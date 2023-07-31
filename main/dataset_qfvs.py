import os
import pdb
import h5py
import nncore
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from nncore.dataset import DATASETS
from nncore.parallel import DataContainer
from main.config_hl import TVSUM_SPLITS, YOUTUBE_SPLITS
from utils.basic_utils import load_jsonl, load_pickle, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from utils.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)

class DatasetQFVS(Dataset):
    def __init__(self,config, use_tef=True):
        # pdb.set_trace()
        self.config=config
        self.dataset=[]
        self.use_tef=use_tef

        self.embedding=load_pickle(f"./data/qfvs/txt_clip/{self.config['txt_feature']}.pkl")
 
        self.transfer={"Cupglass":"Glass",
                  "Musicalinstrument":"Instrument",
                  "Petsanimal":"Animal"}

        self.f_dict = {}
        feat_type = self.config['vid_feature']

        for video_id in self.config["train_videos"]:
            self.f_dict[str(video_id)] = h5py.File(f'./data/qfvs/processed/P0{video_id}_{feat_type}.h5','r')
            for _ , _, files in os.walk("./data/qfvs/metadata/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
                for file in files:
                    self.dataset.append(['Oracle', file[:file.find("_oracle.txt")]+"_"+str(video_id)])
            
            if self.config['qfvs_dense_shot'] > 0:
                dense_concept = {}
                feat_type = self.config['vid_feature']
                feat=h5py.File(f'./data/qfvs/processed/P0{video_id}_{feat_type}.h5','r')
                features=feat['features'][()]
                seg_len=feat['seg_len'][()]
                with open("./data/qfvs/metadata/origin_data/Dense_per_shot_tags/P0"+str(video_id)+"/P0"+str(video_id)+".txt","r") as f:
                        lines=f.readlines()
                        for index,line in enumerate(lines):
                            concepts=line.strip().split(',')
                            for concept in concepts:
                                if concept in self.transfer:
                                   concept= self.transfer[concept]
                                if concept not in dense_concept:
                                    # dense_concept[concept] = torch.zeros(seg_len.sum())
                                    dense_concept[concept] = torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
                                else:
                                    dense_concept[concept][index] = 1

                for key, value in dense_concept.items():
                    if value.sum().item() > 0:
                        self.dataset.append([video_id, key, value])

    def __getitem__(self, index):
        if self.dataset[index][0] == 'Oracle':
            return self.get_oracle(index)
        else:
            return self.get_dense(index)

    def get_dense(self,index):
        video_id=str(self.dataset[index][0])
        f = self.f_dict[video_id]
        # feat_type = self.config['vid_feature']
        # f=h5py.File(f'./data/qfvs/processed/P0{video_id}_{feat_type}.h5','r')
        features=f['features'][()]
        seg_len=f['seg_len'][()]

        dim = features.shape[-1]

        mask_GT = torch.zeros(self.config["max_segment_num"], self.config["max_frame_num"], dtype=torch.bool)
        for j in range(len(seg_len)):
            for k in range(seg_len[j]):
                mask_GT[j][k] = 1

        features = torch.from_numpy(features)

        concept1 = concept2 = self.dataset[index][1]
        concept1_GT = concept2_GT = oracle_summary = self.dataset[index][2]

        if concept1 in self.transfer:
            concept1=self.transfer[concept1]
        if concept2 in self.transfer:
            concept2=self.transfer[concept2]
        concept1=self.embedding[concept1]
        concept2=self.embedding[concept2]

        concept1 = l2_normalize_np_array(concept1)
        concept2 = l2_normalize_np_array(concept2)

        try:
            saliency_pos_labels_1 = torch.Tensor([random.choice(torch.where(concept1_GT> 0)[0].tolist())])
        except:
            saliency_pos_labels_1 = torch.Tensor(0)

        try:
            saliency_pos_labels_2 = torch.Tensor([random.choice(torch.where(concept2_GT> 0)[0].tolist())])
        except:
            saliency_pos_labels_2 = torch.Tensor(0)

        try:
            saliency_pos_labels_oracle = torch.Tensor([random.choice(torch.where(oracle_summary> 0)[0].tolist())])
        except:
            saliency_pos_labels_oracle = torch.Tensor(0)

        return {
            'features':features,
            'seg_len':torch.from_numpy(seg_len),
            'concept1_GT':concept1_GT,
            'concept2_GT':concept2_GT,
            'mask_GT':mask_GT,
            'oracle_summary':oracle_summary,
            'tokens_pad1':torch.from_numpy(concept1),
            'tokens_pad2':torch.from_numpy(concept2),
            'saliency_pos_labels_1': saliency_pos_labels_1,
            'saliency_pos_labels_2': saliency_pos_labels_2,
            'saliency_pos_labels_oracle': saliency_pos_labels_oracle,
        }

    def get_oracle(self,index):
        video_id=self.dataset[index][1].split('_')[2]
        f = self.f_dict[video_id]
        # video_id=self.dataset[index][1].split('_')[2]
        # feat_type = self.config['vid_feature']
        # f=h5py.File(f'./data/qfvs/processed/P0{video_id}_{feat_type}.h5','r')
        features=f['features'][()]
        seg_len=f['seg_len'][()]

        dim = features.shape[-1]

        mask_GT = torch.zeros(self.config["max_segment_num"], self.config["max_frame_num"], dtype=torch.bool)
        for j in range(len(seg_len)):
            for k in range(seg_len[j]):
                mask_GT[j][k] = 1

        features = torch.from_numpy(features)

        concept1,concept2=self.dataset[index][1].split('_')[0:2]

        concept1_GT=torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
        concept2_GT=torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
        # concept1_GT=torch.zeros(seg_len.sum())
        # concept2_GT= torch.zeros(seg_len.sum())
        with open("./data/qfvs/metadata/origin_data/Dense_per_shot_tags/P0"+video_id+"/P0"+video_id+".txt","r") as f:
            lines=f.readlines()
            for index,line in enumerate(lines):
                concepts=line.strip().split(',')
                if concept1 in concepts:
                    concept1_GT[index]=1
                if concept2 in concepts:
                    concept2_GT[index]=1

        # oracle_summary =torch.zeros(seg_len.sum())
        oracle_summary = torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
        GT_summary_shots = []
        with open("./data/qfvs/metadata/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)+"/"+str(concept1)+"_"+str(concept2)+"_"+"oracle.txt","r") as f:
            for line in f.readlines():
                GT_summary_shots.append(int(line.strip()))
        GT_summary_shots = [x - 1 for x in GT_summary_shots]
        for element in GT_summary_shots:
            oracle_summary[element] = 1

        if concept1 in self.transfer:
            concept1=self.transfer[concept1]
        if concept2 in self.transfer:
            concept2=self.transfer[concept2]
        concept1=self.embedding[concept1]
        concept2=self.embedding[concept2]

        concept1 = l2_normalize_np_array(concept1)
        concept2 = l2_normalize_np_array(concept2)

        try:
            saliency_pos_labels_1 = torch.Tensor([random.choice(torch.where(concept1_GT> 0)[0].tolist())])
        except:
            saliency_pos_labels_1 = torch.Tensor(0)

        try:
            saliency_pos_labels_2 = torch.Tensor([random.choice(torch.where(concept2_GT> 0)[0].tolist())])
        except:
            saliency_pos_labels_2 = torch.Tensor(0)

        try:
            saliency_pos_labels_oracle = torch.Tensor([random.choice(torch.where(oracle_summary> 0)[0].tolist())])
        except:
            saliency_pos_labels_oracle = torch.Tensor(0)

        return {
            'features':features,
            'seg_len':torch.from_numpy(seg_len),
            'concept1_GT':concept1_GT,
            'concept2_GT':concept2_GT,
            'mask_GT':mask_GT,
            'oracle_summary':oracle_summary,
            'tokens_pad1':torch.from_numpy(concept1),
            'tokens_pad2':torch.from_numpy(concept2),
            'saliency_pos_labels_1': saliency_pos_labels_1,
            'saliency_pos_labels_2': saliency_pos_labels_2,
            'saliency_pos_labels_oracle': saliency_pos_labels_oracle,
        }

    def __len__(self):
        return len(self.dataset)

def start_end_collate_qfvs(batch):
    model_inputs_keys = batch[0].keys()

    batched_data = dict()
    for k in model_inputs_keys:
        batched_data[k] = pad_sequences_1d([e[k].data for e in batch], dtype=torch.float32, fixed_length=None)

    return batched_data

def prepare_batch_inputs_qfvs(data, config, eval=False):
    if not eval:
        features, mask, seg_len, \
        concept1_GT, concept2_GT, mask_GT, oracle_summary_GT, \
        src_txt_1, src_txt_2, src_txt_mask_1, src_txt_mask_2,\
        saliency_pos_labels_1, saliency_pos_labels_2, saliency_pos_labels_oracle = \
            data['features'][0],  data['mask_GT'][0], data['seg_len'][0],\
            data['concept1_GT'][0], data['concept2_GT'][0], data['mask_GT'][0], data['oracle_summary'][0],\
            data['tokens_pad1'][0], data['tokens_pad2'][0], data['tokens_pad1'][1], data['tokens_pad2'][1], \
            data['saliency_pos_labels_1'][0], data['saliency_pos_labels_2'][0], data['saliency_pos_labels_oracle'][0],
    else:
        features, mask, seg_len, \
        src_txt_1, src_txt_2, src_txt_mask_1, src_txt_mask_2 =  \
            data['features'][0], data['mask_GT'][0], data['seg_len'][0],\
            data['tokens_pad1'][0], data['tokens_pad2'][0], data['tokens_pad1'][1], data['tokens_pad2'][1]

    # preprocess for vid input.
    mask_GT = mask.to('cuda').reshape(1, -1).bool()
    seq = features.to('cuda').squeeze(0)
    mask = mask.to('cuda').squeeze(0)
    num_seg = seq.shape[0]

    ctx_l = seq.shape[1]
    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    tef_ed = tef_st + 1.0 / ctx_l
    tef = torch.stack([tef_st, tef_ed], dim=1).to('cuda')  # (Lv, 2)

    tef = tef.squeeze(0).repeat(seq.shape[0], 1, 1)
    seq = torch.cat([seq, tef], dim=-1)

    # for txt input.
    src_txt_1 = src_txt_1.to(torch.float32).to('cuda').repeat(num_seg, 1, 1)
    src_txt_2 = src_txt_2.to(torch.float32).to('cuda').repeat(num_seg, 1, 1)
    src_txt_mask_1 = src_txt_mask_1.to('cuda').repeat(num_seg, 1)
    src_txt_mask_2 = src_txt_mask_2.to('cuda').repeat(num_seg, 1)

    src_txt_oracle = torch.cat((src_txt_1, src_txt_2), dim=1).to('cuda')
    src_txt_mask_oracle = torch.cat((src_txt_mask_1, src_txt_mask_2), dim=1).to('cuda')

    model_inputs_1 = dict(src_vid=seq, src_vid_mask=mask, src_txt=src_txt_1, src_txt_mask=src_txt_mask_1)
    model_inputs_2 = dict(src_vid=seq, src_vid_mask=mask, src_txt=src_txt_2, src_txt_mask=src_txt_mask_2)
    model_inputs_oracle = dict(src_vid=seq, src_vid_mask=mask, src_txt=src_txt_oracle, src_txt_mask=src_txt_mask_oracle)

    # concept1_GT = concept1_GT.squeeze().reshape(config['max_segment_num'], config['max_frame_num'])
    # concept2_GT = concept2_GT.squeeze().reshape(config['max_segment_num'], config['max_frame_num'])
    # oracle_summary_GT = oracle_summary_GT.squeeze().reshape(config['max_segment_num'], config['max_frame_num'])

    if not eval:
        targets_1 = dict(saliency_scores=concept1_GT.to('cuda'), saliency_pos_labels=saliency_pos_labels_1.to('cuda'))
        targets_2 = dict(saliency_scores=concept2_GT.to('cuda'), saliency_pos_labels=saliency_pos_labels_2.to('cuda'))
        targets_oracle = dict(saliency_scores=oracle_summary_GT.to('cuda'), saliency_pos_labels=saliency_pos_labels_oracle.to('cuda'))

        targets_1['timestamp_mask'] = mask; targets_1['timestamp_window'] = concept1_GT.to('cuda')
        targets_2['timestamp_mask'] = mask; targets_2['timestamp_window'] = concept2_GT.to('cuda')
        targets_oracle['timestamp_mask'] = mask; targets_oracle['timestamp_window'] = oracle_summary_GT.to('cuda')

        return model_inputs_1, model_inputs_2, model_inputs_oracle, \
               targets_1, targets_2, targets_oracle, mask_GT
    else:
        return model_inputs_1, model_inputs_2, model_inputs_oracle, mask_GT