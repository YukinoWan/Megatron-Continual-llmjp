#!/bin/bash

## GPT-3 13B
model_size=13

num_layers=40
hidden_size=5120
num_attn_heads=40

global_batch_size=16

#lr=1.0e-4
lr=3.0e-6
#min_lr=1.0e-6
min_lr=3.3e-7
init_std=0.02

sequence_length=2048

### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens.
train_tokens_in_billion=2
train_tokens=$((${train_tokens_in_billion} * 1000 * 1000 * 1000))

## train_samples is another termination condition and also affect the number of
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
train_samples=$((${train_tokens_in_billion} * 1000000000 / ${sequence_length}))
train_iterations=$((${train_samples}/${global_batch_size}))

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
## Here we increase the warmup tokens to 3B since when batch size warmup is not
## used, there are more tokens per step. Thus we need to increase warmup tokens
## to make sure there are enough warmup steps, which is important for training
## stability.
#lr_warmup_tokens_in_million=3000
#lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
#lr_warmup_samples=1000
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))
lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=1 # tensor model parallel size

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=8
no_pp="false"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=1

## Total number of GPUs
num_gpus_pernode=8
num_node=2
num_gpus=$((${num_gpus_pernode} * ${num_node}))

## Data parallel size.
dp_size=$((${num_gpus} / ${pp_size} / ${mp_size}))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
# batch_size=$(( ${global_batch_size} / ${dp_size} ))
batch_size=1
###############################################################################
### Misc configs
log_interval=1
eval_iters=10
eval_interval=100
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=100
estimated_train_iter=$((${train_tokens} / ${sequence_length} / ${global_batch_size}))
# save_interval=$((${estimated_train_iter} / ${num_save}))
save_interval=3000

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=123
num_workers=2


export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config

# distributed settings
export MASTER_ADDR=10.4.252.59
export MASTER_PORT=6000

echo "MASTER_ADDR=${MASTER_ADDR}"

NODE_TYPE="a100"
export NUM_GPU_PER_NODE=8

NUM_NODES=${num_node}
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO

HOSTFILE_NAME=/home/zhen/Megatron-JaMedLLM/hostfile/hostfile_2node


# job name
jobname="gpt_${model_size}B_token${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_gpu${num_gpus}"


if [[ $mp_size -gt 1 ]]; then
  jobname="${jobname}_tp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
  jobname="${jobname}_pp${pp_size}"
fi


# output dir
output_home="/large/wan/megatronlm-medllm-same-ratio-warm0.03-outputs"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
#CHECKPOINT_PATH="${output_home}/checkpoint/${jobname}"
model_path="/home/zhen/Megatron-DeepSpeed/models/llm-jp-tokenizer/models/ver2.1/code10k_en20k_ja30k.ver2.1.model"
## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="${output_home}/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
DATA_CACHE_PATH="${output_home}/data_cache/${jobname}"


# dataset
data_ratio_1="997290277"
data_path_1="/fast/wan/med_train/eng_text_document"
data_ratio_2="429673921"
data_path_2="/fast/wan/med_train/ja_text_document"

data_path="${data_ratio_1} ${data_path_1} ${data_ratio_2} ${data_path_2}"
#data_path="/fast/wan/train_bilingual/bilingual_text_document"

DROP_OUT=0.0
NUM_QUERY_GROUP=40
FFN_HIDDEN_SIZE=13824
NORM_EPS=1e-5

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

megatron_args=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --pipeline-model-parallel-size ${pp_size} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${sequence_length} \
    --max-position-embeddings ${sequence_length} \
    --train-iters ${train_iterations} \
    --lr-warmup-fraction 0.03 \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50f,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --num-workers ${num_workers} \
    --distributed-backend nccl \
    --bf16 \
    --seed ${seed} \
    --load /large/wan/models/llm-jp-13b-v1.0-megatron-lm-tp1-pp8 \
    --finetune \
    --no-load-optim \
    --no-load-rng \
    --save ${checkpoint_path} \
    --use-flash-attn \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path} \
    --distributed-timeout-minutes 3600 \
    --recompute-activations \
    --recompute-granularity "selective" \
"
DATA_ARGS="
    --tokenizer-model ${model_path} \
    --tokenizer-type SentencePieceTokenizer \
    --data-path ${data_path}\
"

export NCCL_DEBUG=INFO

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
  -bind-to none -map-by slot \
  -x PATH \
   python pretrain_gpt.py \
    $megatron_args \
    $DATA_ARGS \
    --use-mpi \
    --wandb-project "megatronlm-llmjp13b-med-same-ratio" \
    --wandb-exp-name "${jobname}"\
    --wandb-save-dir "outputs/wandb/${jobname}" \
    --distributed-backend nccl \
    &>>${log_path}/${jobname}_${host}_${current_time}.log
