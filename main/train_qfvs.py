import os
import pdb
import time
import json
import pprint
import random
import importlib
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import h5py
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/Users/kevin/univtg')
from main.config import BaseOptions, setup_model
from main.dataset_qfvs import DatasetQFVS, prepare_batch_inputs_qfvs, start_end_collate_qfvs
from utils.basic_utils import set_seed, AverageMeter, dict_to_markdown, save_json, save_jsonl, load_json, load_pickle, l2_normalize_np_array
from utils.model_utils import count_parameters
from eval.qfvs import calculate_semantic_matching, load_videos_tag

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def eval_epoch(model, config, opt):
    model.eval()
    f1_sum = 0; p_sum = 0; r_sum = 0

    assert len(config['test_videos']) == 1
    video_id = config['test_videos'][0]
    embedding = load_pickle(f"./data/qfvs/txt_clip/{config['txt_feature']}.pkl")

    feat_type = config['vid_feature']
    feat = h5py.File(f'./data/qfvs/processed/P0{video_id}_{feat_type}.h5', 'r')
    features = torch.from_numpy(feat['features'][()])
    seg_len = torch.from_numpy(feat['seg_len'][()])
    # seg_len = torch.tensor(feat['seg_len'][()]).unsqueeze(0).cuda()
  
    # dim = features.shape[-1]
    # ctx_l = seg_len.sum().cpu()

    # dim = features.shape[-1]
    # ctx_l =   features.shape[1]
    # seg_len = torch.ones(ctx_l)
    # features = features.reshape(-1, dim)[:ctx_l]

    # tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    # tef_ed = tef_st + 1.0 / ctx_l
    # tef = torch.stack([tef_st, tef_ed], dim=1).cuda() # (Lv, 2)
    # features = torch.cat([features, tef], dim=1)  # (Lv, Dv+2)

    transfer = {"Cupglass": "Glass",
                "Musicalinstrument": "Instrument",
                "Petsanimal": "Animal"}

    for _,_,files in os.walk("./data/qfvs/metadata/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
        evaluation_num=len(files)

        mask_GT = torch.zeros(config["max_segment_num"], config["max_frame_num"], dtype=torch.bool).cuda()
        for j in range(len(seg_len)):
            for k in range(seg_len[j]):
                mask_GT[j][k] = 1

        for file in files:
            summaries_GT=[]
            with open("./data/qfvs/metadata/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)+"/"+file,"r") as f:
                for line in f.readlines():
                    summaries_GT.append(int(line.strip()))

            concept1, concept2 = file.split('_')[0:2]

            ##############
            if concept1 in transfer:
                concept1 = transfer[concept1]
            if concept2 in transfer:
                concept2 = transfer[concept2]
            concept1 = embedding[concept1]
            concept2 = embedding[concept2]

            concept1 = l2_normalize_np_array(concept1)
            concept2 = l2_normalize_np_array(concept2)

            data = {
            'features':features,
            'seg_len': seg_len,
            'tokens_pad1':torch.from_numpy(concept1),
            'tokens_pad2':torch.from_numpy(concept2),
            'mask_GT': mask_GT
            }

            input1, input2, input_oracle, mask = prepare_batch_inputs_qfvs(start_end_collate_qfvs([data]), config, eval=True)

            summaries_GT = [x - 1 for x in summaries_GT]
            video_shots_tag = load_videos_tag(mat_path="./eval/Tags.mat")

            if opt.f_loss_coef == 0:
                output_type = 'saliency_scores'
            elif opt.s_loss_intra_coef == 0:
                output_type = 'pred_logits'
            else:
                if config['qfvs_score_ensemble'] > 0:
                    output_type = ['pred_logits', 'saliency_scores']
                else:
                    output_type = 'pred_logits'

            with torch.no_grad():
                if not isinstance(output_type, list):
                    score1 = model(**input1)[output_type].squeeze()
                    score1 = score1.masked_select(mask_GT)

                    score2 = model(**input2)[output_type].squeeze()
                    score2 = score2.masked_select(mask_GT)

                    score = model(**input_oracle)[output_type].squeeze()
                    score = score.masked_select(mask_GT)
                else:
                    score1, score2, score = torch.zeros((int(mask.sum().item()))).cuda(),  torch.zeros((int(mask.sum().item()))).cuda(),  torch.zeros((int(mask.sum().item()))).cuda()
                    for output_t in output_type:
                        score1 += model(**input1)[output_t].squeeze().masked_select(mask_GT)
                        score2 += model(**input2)[output_t].squeeze().masked_select(mask_GT)
                        score += model(**input_oracle)[output_t].squeeze().masked_select(mask_GT)

                if config['qfvs_score_gather'] > 0:
                    score = score + score1 + score2
                else:
                    score = score

                # since video4 features dim is greater than video_shots_tag.
                score = score[:min(score.shape[0], video_shots_tag[video_id-1].shape[0])]
                _, top_index = score.topk(int(score.shape[0] * config["top_percent"]))

                p, r, f1 = calculate_semantic_matching(list(top_index.cpu().numpy()), summaries_GT, video_shots_tag, video_id=video_id-1)
                f1_sum+=f1;  r_sum+=r; p_sum+=p

    return {'F': round(100* f1_sum/evaluation_num,2) ,
            'R': round(100* r_sum/evaluation_num,2) ,
            'P': round(100* p_sum/evaluation_num,2) }

def idx2time(idx):
    sec1, sec2 = idx*5, (idx+1)*5

    h1 = sec1 // 3600
    m1 = (sec1 - h1*3600) // 60
    s1 = sec1 % 60

    h2 = sec2 // 3600
    m2 = (sec2 - h2*3600) // 60
    s2 = sec2 % 60
    print(h1,m1,s1,'\t', h2,m2,s2)

def train_epoch(model, criterion, train_loader, optimizer, opt, config, epoch_i, tb_writer):
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    timer_dataloading = time.time()
    loss_total = 0

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)
        timer_start = time.time()
        model_input1, model_input2, model_input_oracle, \
        model_gt1, model_gt2, model_gt_oracle, \
        mask_GT = prepare_batch_inputs_qfvs(batch, config)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        timer_start = time.time()
        output1 = model(**model_input1)
        output2 = model(**model_input2)
        output_oracle = model(**model_input_oracle)

        loss_dict = {}
        loss_dict1 = criterion(output1, model_gt1, mask_GT)
        loss_dict2 = criterion(output2, model_gt2, mask_GT)
        loss_dict3 = criterion(output_oracle, model_gt_oracle, mask_GT)

        weight_dict = criterion.weight_dict
        if config['qfvs_loss_gather'] > 0:
            for k in loss_dict1.keys():
                loss_dict[k] = loss_dict1[k] + loss_dict2[k] + loss_dict3[k]
        else:
            loss_dict = loss_dict3

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_total += losses.item()

        time_meters["model_forward_time"].update(time.time() - timer_start)
        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        timer_dataloading = time.time()
    return round(loss_total  / len(train_loader), 2)

# train in single domain.
def train(model, criterion, optimizer, lr_scheduler, train_loader, opt, config):
    # if opt.device.type == "cuda":
        # logger.info("CUDA enabled.")
        # model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    prev_best_score = {'Fscore':0, 'Precision':0, 'Recall':0}
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_init else 0
    else:
        start_epoch = opt.start_epoch

    val_score = eval_epoch(model, config, opt)
    tb_writer.add_scalar(f"Eval/QFVS-V{config['test_videos'][0]}-fscore", float(val_score['F']), 0)
    logger.info(f"[Epoch {0}] [Fscore: {val_score['F']} / {prev_best_score['Fscore']}]"
                f" [Precision: {val_score['P']} / {prev_best_score['Precision']}]"
                f" [Recall: {val_score['R']} / {prev_best_score['Recall']}]")
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            loss_epoch = train_epoch(model, criterion, train_loader, optimizer, opt, config, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = opt.eval_epoch
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                val_score = eval_epoch(model, config, opt)
                tb_writer.add_scalar(f"Eval/QFVS-V{config['test_videos'][0]}-fscore", float(val_score['F']), epoch_i+1)
            logger.info(f"[Epoch {epoch_i + 1}, Loss {loss_epoch}] [Fscore: {val_score['F']} / {prev_best_score['Fscore']}]"
                        f" [Precision: {val_score['P']} / {prev_best_score['Precision']}]"
                        f" [Recall: {val_score['R']} / {prev_best_score['Recall']}]")

            if prev_best_score['Fscore'] < val_score['F']:
                prev_best_score['Fscore'] = val_score['F']
                prev_best_score['Precision'] = val_score['P']
                prev_best_score['Recall'] = val_score['R']

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_V{config['test_videos'][0]}_best.ckpt"))
    tb_writer.close()
    return prev_best_score

def update_config(opt, config):
    # for key in ["max_segment_num", "max_frame_num", "top_percent", 
                # "qfvs_vid_feature", "qfvs_txt_feature", "qfvs_dense_shot",
                # "qfvs_score_ensemble", "qfvs_score_gather", "qfvs_loss_gather"]:
    config["max_segment_num"] = opt.max_segment_num
    config["max_frame_num"] = opt.max_frame_num
    config["top_percent"] = opt.top_percent
    config["vid_feature"] = opt.qfvs_vid_feature
    config["txt_feature"] = opt.qfvs_txt_feature
    config["qfvs_dense_shot"] = opt.qfvs_dense_shot
    config["qfvs_score_ensemble"] = opt.qfvs_score_ensemble
    config["qfvs_score_gather"] = opt.qfvs_score_gather
    config["qfvs_loss_gather"] = opt.qfvs_loss_gather
    return config

def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)

    # config = load_json("./main/config_qfvs.json")
    config = {}
    config = update_config(opt, config)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)

    # key -> test video; value -> training videos.
    qfvs_split = {
                    1: [2, 3, 4],
                  2: [1, 3, 4],
                  3: [1, 2, 4],
                  4: [1, 2, 3]
                  }

    scores_videos = {}
    for test_id, splits in qfvs_split.items():
        logger.info(f"Start Training {opt.dset_name}: {test_id}")
        config['train_videos'] = qfvs_split[test_id]
        config['test_videos'] = [test_id]
        train_dataset = DatasetQFVS(config)
        train_loader = DataLoader(train_dataset, batch_size=opt.bsz, collate_fn=start_end_collate_qfvs, shuffle=True, num_workers=opt.num_workers)

        model, criterion, optimizer, lr_scheduler = setup_model(opt)
        count_parameters(model)
        best_score = train(model, criterion, optimizer, lr_scheduler, train_loader, opt,  config)
        scores_videos['V'+str(test_id)] = best_score

    # save the final results.
    avg_fscore = sum([v['Fscore'] for k, v in scores_videos.items()]) / len(scores_videos)
    avg_precision = sum([v['Precision'] for k, v in scores_videos.items()]) / len(scores_videos)
    avg_recall = sum([v['Recall'] for k, v in scores_videos.items()]) / len(scores_videos)
    scores_videos['avg'] = {'Fscore':avg_fscore, 'Precision':avg_precision, 'Recall':avg_recall}

    save_metrics_path = os.path.join(opt.results_dir, f"best_{opt.dset_name}_{opt.eval_split_name}_preds_metrics.json")
    save_json( scores_videos, save_metrics_path, save_pretty=True, sort_keys=False)

    tb_writer.add_scalar(f"Eval/QFVS-avg-fscore", round(avg_fscore, 2), 1)
    tb_writer.add_text(f"Eval/QFVS-{opt.dset_name}", dict_to_markdown(scores_videos, max_str_len=None))
    tb_writer.close()

    print(scores_videos)
    return

if __name__ == '__main__':
    start_training()
    results = logger.info("\n\n\nFINISHED TRAINING!!!")
