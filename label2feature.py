import pdb
import sys
import json
import torch
import numpy as np
from run_on_video.data_utils import ClipFeatureExtractor
import torch.nn.functional as F
import tqdm
import os

for index in [0]:
    json_pool_list = {
        0:'/teacher/oidv6-class-descriptions.json'
    }
    json_path = json_pool_list[index]
    save_dir = f'/data/commonsense/clip_txt/{index}.npz'

    with open(json_path,'r') as load_f:
        query_list = json.load(load_f)
    query_list = [f'a photo of a {x}' for x in query_list]
    # pdb.set_trace()

    qid_list = list(range(len(query_list)))

    # clip
    feature_extractor = ClipFeatureExtractor(
        framerate=1 / 2, size=224, centercrop=True,
        model_name_or_path="ViT-B/32", device='cuda'
    )

    query_feats, query_feats_pooler = feature_extractor.encode_text(query_list, pooler=True)
    query_feats_pooler = torch.concat([x.unsqueeze(0) for x in query_feats_pooler], 0).cpu().numpy()

    np.savez(save_dir, pooler_output=query_feats_pooler)
