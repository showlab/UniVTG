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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
from main.config import BaseOptions, setup_model
from main.dataset import DatasetHL, prepare_batch_inputs_hl, start_end_collate_hl
from utils.basic_utils import set_seed, AverageMeter, dict_to_markdown, save_json, save_jsonl
from utils.model_utils import count_parameters

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def eval_epoch(model, train_val_dataset, opt): #, nms_thresh, device):
    model.eval()

    scores = []
    train_val_dataset.set_state('val')
    val_loader = DataLoader(
        train_val_dataset,
        collate_fn=start_end_collate_hl,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    with torch.no_grad():
        for data in val_loader:
            model_inputs, targets = prepare_batch_inputs_hl(data)
            outputs = model(**model_inputs)
            # pred_cls = outputs['pred_logits'].squeeze(-1)
            # pred_cls = outputs['saliency_scores']
            # pred_cls = outputs['saliency_scores'] + outputs['pred_logits'].squeeze(-1)

            # pdb.set_trace()
            if opt.f_loss_coef == 0:
                pred_cls = outputs['saliency_scores']
            elif opt.s_loss_intra_coef == 0:
                pred_cls = outputs['pred_logits'].squeeze(-1)
            else:
                if opt.eval_mode == 'add':
                    pred_cls = outputs['saliency_scores'] + outputs['pred_logits'].squeeze(-1)
                else:
                    pred_cls = outputs['pred_logits'].squeeze(-1)

            pred_cls = pred_cls.detach().cpu()
            scores.append(pred_cls)
        map = round(train_val_dataset.evaluate(scores, save_dir='./plot')['mAP'] * 100,  4)
    return map

def train_epoch(model, criterion, train_val_dataset, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    train_val_dataset.set_state('train')
    train_loader = DataLoader(
        train_val_dataset,
        collate_fn=start_end_collate_hl,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in enumerate(train_loader):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)
        timer_start = time.time()
        model_inputs, targets = prepare_batch_inputs_hl(batch)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        timer_start = time.time()
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")

# train in single domain.
def train(model, criterion, optimizer, lr_scheduler, train_val_dataset, opt):
    # if opt.device.type == "cuda":
        # logger.info("CUDA enabled.")
        # model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    prev_best_score = 0.
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_init else 0
    else:
        start_epoch = opt.start_epoch

    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_val_dataset, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = opt.eval_epoch
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                scores = eval_epoch(model, train_val_dataset, opt)
                tb_writer.add_scalar(f"Eval/HL-{opt.dset_name}-{train_val_dataset.domain}-mAP", float(scores), epoch_i+1)
            if prev_best_score < scores:
                prev_best_score = scores
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_{train_val_dataset.domain}_best.ckpt"))
    tb_writer.close()
    return prev_best_score

def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)

    from main.config_hl import TVSUM_SPLITS, YOUTUBE_SPLITS
    if opt.dset_name == "tvsum":
        domain_splits = TVSUM_SPLITS.keys()
    if opt.dset_name == "youtube":
        domain_splits = YOUTUBE_SPLITS.keys()

    scores = {}
    if opt.lr_warmup > 0:
        # total_steps = opt.n_epoch * len(train_dataset) // opt.bsz
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]
    
    domain_splits = domain_splits if not opt.domain_name else [opt.domain_name]

    for domain in domain_splits:
        dataset_config = dict(
            dset_name=opt.dset_name,
            domain=domain,
            data_path=opt.train_path,
            v_feat_types=opt.v_feat_types,
            v_feat_dirs=opt.v_feat_dirs,
            t_feat_dir=opt.t_feat_dir,
            use_tef=True
        )
        dataloader = DatasetHL(**dataset_config)

        model, criterion, optimizer, lr_scheduler = setup_model(opt)
        count_parameters(model)
        logger.info(f"Start Training {domain}")
        best_score = train(model, criterion, optimizer, lr_scheduler, dataloader, opt)
        scores[domain] = best_score
    scores['AVG'] = sum(scores.values()) / len(scores)

    # save the final results.
    save_metrics_path = os.path.join(opt.results_dir, f"best_{opt.dset_name}_{opt.eval_split_name}_preds_metrics.json")
    save_json(scores, save_metrics_path, save_pretty=True, sort_keys=False)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text(f"HL-{opt.dset_name}", dict_to_markdown(scores, max_str_len=None))
    tb_writer.add_scalar(f"Eval/HL-{opt.dset_name}-avg-mAP-key", float(scores['AVG']), 1)
    tb_writer.close()
    # return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug

    print(opt.dset_name)
    print(scores)
    return

if __name__ == '__main__':
    start_training()
    results = logger.info("\n\n\nFINISHED TRAINING!!!")
