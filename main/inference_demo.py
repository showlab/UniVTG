import pdb
import pprint
from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from main.config import TestOptions, setup_model
from main.dataset import DatasetMR, start_end_collate_mr, prepare_batch_inputs_mr
from eval.eval import eval_submission
from eval.postprocessing import PostProcessorDETR
from utils.basic_utils import save_jsonl, save_json
from utils.temporal_nms import temporal_nms
from utils.span_utils import span_cxw_to_xx
from utils.basic_utils import load_jsonl, load_pickle, l2_normalize_np_array

import logging
import importlib

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def load_model():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    # pdb.set_trace()
    cudnn.benchmark = True
    cudnn.deterministic = False

    model, criterion, _, _ = setup_model(opt)
    return model

def load_data(save_dir):
  vid = np.load(os.path.join(save_dir, 'vid.npz'))['features'].astype(np.float32)
  txt = np.load(os.path.join(save_dir, 'txt.npz'))['features'].astype(np.float32)

  vid = torch.from_numpy(l2_normalize_np_array(vid))
  txt = torch.from_numpy(l2_normalize_np_array(txt))
  clip_len = 2
  ctx_l = vid.shape[0]

  timestamp =  ( (torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

  if True:
      tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
      tef_ed = tef_st + 1.0 / ctx_l
      tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
      vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

  src_vid = vid.unsqueeze(0).cuda()
  src_txt = txt.unsqueeze(0).cuda()
  src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
  src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

  return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

if __name__ == '__main__':
    clip_len = 2
    save_dir = '/data/home/qinghonglin/univtg/demo/tmp'
  
    model = load_model()
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(save_dir)
    with torch.no_grad():
      output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)

    pred_logits = output['pred_logits'][0].cpu()
    pred_spans = output['pred_spans'][0].cpu()
    pred_saliency = output['saliency_scores'].cpu()

    pdb.set_trace()
    top1 = (pred_spans + timestamp)[torch.argmax(pred_logits)] * ctx_l * clip_len
    print(top1)
    print(pred_saliency.argmax()*clip_len)