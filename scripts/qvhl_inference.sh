#!/bin/bash
#SBATCH --job-name=qvhl
#SBATCH --output=/fsx/qinghonglin/univtg/log/qvhl_ft.log
#SBATCH --partition=learnai4rl

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10 
#SBATCH --account all

export NCCL_SOCKET_IFNAME=ens32
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2  

dset_type=mr
dset_name=qvhighlights
clip_length=2

gpu_id=0
num_workers=16

exp_id=qvhl
model_id=univtg

bsz=32
eval_bsz=4
n_epoch=0
lr=1e-4
lr_drop=80
lr_warmup=10
wd=1e-4

input_dropout=0.5
dropout=0
droppath=0.1

eval_epoch=5
enc_layers=4
eval_mode=add
round_multiple=-1
hidden_dim=1024

b_loss_coef=10
g_loss_coef=0
eos_coef=0.1
f_loss_coef=10
s_loss_intra_coef=0
s_loss_inter_coef=0

main_metric=MR-full-mAP-key
nms_thd=0.7
max_before_nms=1000

ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
use_cache=1
easy_negative_only=1

resume={}

# kill pid in gpu_id
# ps -up `nvidia-smi -i ${gpu_id} -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//'` | awk '{print "kill -9 " $2;}' | sh

######## data paths
train_path=data/${dset_name}/metadata/qvhighlights_train.jsonl
eval_path=data/${dset_name}/metadata/qvhighlights_val.jsonl
eval_split_name=val
feat_root=data/${dset_name}

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_slowfast)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"i3d"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_i3d)
  (( v_feat_dim += 1024 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"c3d"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_c3d)
  (( v_feat_dim += 500 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_clip)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/txt_clip
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

srun --label python3 ./main/train_mr.py \
--dset_type ${dset_type}   \
--dset_name ${dset_name} \
--clip_length ${clip_length} \
--exp_id ${exp_id} \
--gpu_id ${gpu_id} \
--model_id ${model_id} \
--v_feat_types ${v_feat_types} \
--t_feat_type ${t_feat_type} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
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
--n_epoch ${n_epoch} \
--num_workers ${num_workers} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--lr_warmup ${lr_warmup} \
--wd ${wd} \
--use_cache ${use_cache} \
--enc_layers ${enc_layers} \
--main_metric ${main_metric} \
--nms_thd ${nms_thd} \
--easy_negative_only ${easy_negative_only} \
--max_before_nms ${max_before_nms} \
--b_loss_coef ${b_loss_coef} \
--g_loss_coef ${g_loss_coef} \
--eos_coef ${eos_coef} \
--f_loss_coef ${f_loss_coef} \
--s_loss_intra_coef ${s_loss_intra_coef}  \
--s_loss_inter_coef ${s_loss_inter_coef} \
--eval_mode ${eval_mode} \
--round_multiple ${round_multiple} \
--hidden_dim ${hidden_dim} \
--eval_init ${@:1}  \
--resume ${resume}  \
