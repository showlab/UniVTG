#!/bin/bash
#SBATCH --job-name=omni
#SBATCH --output=/fsx/qinghonglin/univtg/log/omni.log

#SBATCH --partition=learnai4rl

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --account all

export NCCL_SOCKET_IFNAME=ens32
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

dset_type=vlp
dset_name=vlp
clip_length=2

exp_id=omni_aio_unified__epo6_f10_b10g1_s0.1_0.1
model_id=univtg
gpu_id=0
num_workers=8

bsz=64
eval_bsz=32
n_epoch=100
lr=1e-4
lr_warmup=10
lr_drop=200
wd=1e-4

input_dropout=0.5
dropout=0
droppath=0.1

eval_epoch=5
enc_layers=4
save_interval=5

b_loss_coef=10
g_loss_coef=1
eos_coef=0.1
f_loss_coef=10
s_loss_intra_coef=0.1
s_loss_inter_coef=0.1
hidden_dim=1024

main_metric=MR-full-R1@0.3-key
nms_thd=0.7
max_before_nms=1000

ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
use_cache=-1
easy_negative_only=1
add_easy_negative=1

resume=/data/home/qinghonglin/univtg/results/vlp-vlp/aio_unified-slowfast_clip-clip-2023_05_26_07/model_e0006.ckpt

######## data paths
train_path=()
train_path+=(data/qvhighlights/metadata/qvhighlights_train.jsonl)
train_path+=(data/charades/metadata/charades_train.jsonl)
train_path+=(data/ego4d/metadata/nlq_train.jsonl)
train_path+=(data/tacos/metadata/train.jsonl)
train_path+=(data/anet/metadata/train.jsonl)
train_path+=(data/didemo/metadata/train.jsonl)


eval_path=data/qvhighlights/metadata/qvhighlights_val.jsonl
eval_split_name=val

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(vid_slowfast)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"i3d"* ]]; then
  v_feat_dirs+=(vid_i3d)
  (( v_feat_dim += 1024 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"c3d"* ]]; then
  v_feat_dirs+=(vid_c3d)
  (( v_feat_dim += 500 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(vid_clip)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=txt_clip
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

srun --label python -m torch.distributed.launch --nproc_per_node=4 --max_restarts=0 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 ./main/train_vlp_ddp.py \
--dset_type ${dset_type}    \
--dset_name ${dset_name}    \
--clip_length ${clip_length}    \
--exp_id ${exp_id} \
--gpu_id ${gpu_id} \
--model_id ${model_id} \
--v_feat_types ${v_feat_types} \
--t_feat_type ${t_feat_type} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path[@]} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--eval_epoch ${eval_epoch} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--input_dropout ${input_dropout} \
--dropout ${dropout} \
--droppath ${droppath} \
--bsz ${bsz} \
--eval_bsz ${eval_bsz} \
--save_interval ${save_interval} \
--n_epoch ${n_epoch} \
--num_workers ${num_workers} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--lr_warmup ${lr_warmup}  \
--wd ${wd} \
--use_cache ${use_cache} \
--enc_layers ${enc_layers} \
--main_metric ${main_metric} \
--nms_thd ${nms_thd} \
--easy_negative_only ${easy_negative_only} \
--add_easy_negative ${add_easy_negative} \
--max_before_nms ${max_before_nms} \
--b_loss_coef ${b_loss_coef} \
--g_loss_coef ${g_loss_coef} \
--eos_coef ${eos_coef} \
--f_loss_coef ${f_loss_coef} \
--s_loss_intra_coef ${s_loss_intra_coef}  \
--s_loss_inter_coef ${s_loss_inter_coef} \
--hidden_dim ${hidden_dim} \
--resume ${resume} \
--eval_init \
${@:1}
