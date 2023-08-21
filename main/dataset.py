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
from random import shuffle

logger = logging.getLogger(__name__)

class DatasetVLP(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """
    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir, v_feat_dim, q_feat_dim,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 use_cache=-1, fix_len=-1, add_easy_negative=1, easy_negative_only=-1):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.v_feat_dim = v_feat_dim
        self.q_feat_dim = q_feat_dim
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.fix_len = fix_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.use_cache = use_cache
        self.add_easy_negative = add_easy_negative
        self.easy_negative_only = easy_negative_only

        self.vlp_mapping = {
            # pre-training
            'data/ego4d/metadata/point_egoclip_wo_val.jsonl': {      
                'dset_name': 'ego4d', 'v_feat_suffix': '_point', 'q_feat_suffix': '_point', 'type': 'point',
            },
            'data/videocc/metadata/interval_900k.jsonl': {      
                'dset_name': 'videocc', 'v_feat_suffix': '', 'q_feat_suffix': '', 'type': 'interval',
            },
            'data/videocc/metadata/curve_5_window.jsonl': {      
                'dset_name': 'videocc', 'v_feat_suffix': '', 'q_feat_suffix': '_concept', 'type': 'curve',
            },
            # downstream
            'data/qvhighlights/metadata/qvhighlights_train.jsonl': {      
                'dset_name': 'qvhighlights', 'v_feat_suffix': '', 'q_feat_suffix': '', 'type': 'curve',
            },
            'data/charades/metadata/charades_train.jsonl': {      
                # 'dset_name': 'charades', 'v_feat_suffix': '_2', 'q_feat_suffix': '', 'type': 'interval',
                'dset_name': 'charades', 'v_feat_suffix': '', 'q_feat_suffix': '', 'type': 'interval',
            },
            'data/ego4d/metadata/nlq_train.jsonl': {
                'dset_name': 'ego4d', 'v_feat_suffix': '', 'q_feat_suffix': '', 'type': 'interval',
            },
            'data/tacos/metadata/train.jsonl': {      
                'dset_name': 'tacos', 'v_feat_suffix': '', 'q_feat_suffix': '', 'type': 'interval',
            },
            'data/anet/metadata/train.jsonl': {      
                'dset_name': 'anet', 'v_feat_suffix': '', 'q_feat_suffix': '', 'type': 'interval',
            },
            'data/didemo/metadata/train.jsonl': {      
                'dset_name': 'didemo', 'v_feat_suffix': '', 'q_feat_suffix': '', 'type': 'interval',
            },
        }


        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

        self.v_feat_types = [feat_dir.split('/')[-1] for feat_dir in self.v_feat_dirs]
        t_feat_type = q_feat_dir.split('/')[-1]

        if self.use_cache > 0:
            print('Loading the off-line features...')
            dset_dir = os.path.join('data', self.dset_name)
            vid_keys = [meta['vid'] for meta in self.data]
            qid_keys = [meta['qid'] for meta in self.data]

            self.vid_cache = {}
            for v_feat_type in self.v_feat_types:
                assert 'vid' in v_feat_type
                with h5py.File(os.path.join(dset_dir, 'h5py', v_feat_type + '.hdf5'), 'r') as f:
                    self.vid_cache[v_feat_type] = {key: f[str(key)][:] for key in tqdm(vid_keys)}

            assert 'txt' in t_feat_type
            self.txt_cache = {}
            with h5py.File(os.path.join(dset_dir, 'h5py', t_feat_type + '.hdf5'), 'r') as f:
                for key in tqdm(qid_keys):
                    try:
                        self.txt_cache[key] = f[str(key)][:]
                    except:
                        logger.info(f"text {key} is not in the cache.")

    def load_data(self):
        # datalist = load_jsonl(self.data_path[0])
        datalist = []
        for dset_path in self.data_path:
            dset_info = self.vlp_mapping[dset_path]
            dset_list = load_jsonl(dset_path)
            for x in dset_list: x.update(dset_info)
            datalist += dset_list
        n_examples = int(len(datalist))
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            shuffle(datalist)
            datalist = datalist[:n_examples]
        logger.info("Using {}% of the data: {} examples"
            .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta)  # (Dq, ) or (Lq, Dq)

        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta)  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if meta['dset_name'] in ['hacs', 'ego4d', 'activitynet']:
            for i, window_i in enumerate(meta["relevant_windows"]):
                if window_i[1] - window_i[0] < self.clip_len:
                    center = (window_i[1] + window_i[0]) / 2 
                    window_i[0] = max(0, center - 0.5 * self.clip_len)
                    window_i[1] = min(float(meta['duration']), center + 0.5 * self.clip_len)
                    window_i[1] = max(self.clip_len, window_i[1])

        model_inputs["timestamp"] = ( (torch.arange(0, ctx_l) + self.clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

        if 'test' in self.data_path and 'qvhighlights' in self.dset_name:
            meta["relevant_windows"] = [[0, 150]]
        relevant_windows = torch.Tensor(meta["relevant_windows"])

        # assign the nearest window for each timestamp i.e., qvhighlights.
        num_vid_seq = model_inputs["timestamp"].shape[0]
        num_windows = relevant_windows.shape[0]

        relevant_windows_ts = relevant_windows / (ctx_l * self.clip_len)
        relevant_windows_ts = relevant_windows_ts.unsqueeze(0).repeat(num_vid_seq, 1, 1)
        model_inputs_ts = model_inputs["timestamp"].unsqueeze(1).repeat(1, num_windows, 1)

        if meta['qid'] is not None:
            nn_window_ts = torch.zeros_like(model_inputs["timestamp"])
            diff_left = model_inputs_ts[..., 0]  - relevant_windows_ts[..., 0]
            diff_right = relevant_windows_ts[..., 1] - model_inputs_ts[..., 1]
            assign_idx = torch.where((diff_left >= 0) * (diff_right >= 0))
            if min(assign_idx[0].shape) == 0:   # not assigned, happened in activitynet.
                nn_window_ts = relevant_windows_ts.squeeze(1)
            else:
                nn_window_ts[assign_idx[0]] = relevant_windows_ts[assign_idx[0], assign_idx[1]]

        model_inputs["span_labels_nn"] = nn_window_ts
        model_inputs["timestamp_window"] = 1 * (model_inputs["timestamp"][:,0] >= nn_window_ts[:,0])  & (model_inputs["timestamp"][:,1] <= nn_window_ts[:,1])

        # for activitynet.
        if model_inputs["timestamp_window"].sum() < 1:
            idx = int(meta['relevant_windows'][0][0] / self.clip_len)
            idx = max(0, min(idx, ctx_l-1))
            model_inputs["timestamp_window"][idx] = 1

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
            if 'saliency_scores' in meta.keys():
                # this is for highlight-only task
                model_inputs["saliency_scores"] = torch.zeros(ctx_l).double()
                limit = meta["relevant_clip_ids"].index(ctx_l) if (np.array(meta["relevant_clip_ids"]) >= ctx_l).any() else None
                model_inputs["saliency_scores"][meta["relevant_clip_ids"][:limit]] = torch.tensor(np.mean(np.array(meta["saliency_scores"][:limit]), -1))
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
                # pdb.set_trace()
            else:
                model_inputs["saliency_scores"] = model_inputs["timestamp_window"]
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)  # only one gt
                model_inputs["saliency_pos_labels"] = [ random.choice(torch.where(model_inputs['saliency_scores'])[0].tolist()) ]

        if 'type' in meta.keys():
            if meta['type'] == 'point':
                model_inputs['weight_ablation'] = torch.tensor([0, 0, 1, 0, 0])
            if meta['type'] == 'interval':
                model_inputs['weight_ablation'] = torch.tensor([1, 1, 0, 0, 0])
            if meta['type'] == 'curve':
                model_inputs['weight_ablation'] = torch.tensor([0, 0, 0, 1, 1])

        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=1):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_st = min(gt_st, ctx_l-1)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            # gt_st = gt_ed
            gt_ed = gt_st

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st] * max_n #[gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        # neg_clip_indices = random.sample(neg_pool, k=max_n)

        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]

        if agg_scores[sort_indices[-1]] == agg_scores[sort_indices[0]]:
            hard_neg_clip_indices = hard_pos_clip_indices

        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        # pdb.set_trace()
        if self.add_easy_negative > 0:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        if self.easy_negative_only > 0:
            return easy_pos_clip_indices, easy_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, meta):
        qid = meta['qid']
        dset_name = meta['dset_name']
        q_feat_suffix = meta['q_feat_suffix']
        q_feat_dir = self.q_feat_dir +  q_feat_suffix

        if self.use_cache > 0:
            try:
                q_feat = self.txt_cache[qid]
            except:
                q_feat = np.zeros((10, self.q_feat_dim)).astype(np.float32)
            return  torch.from_numpy(q_feat)

        q_feat_path = os.path.join('data', dset_name, q_feat_dir, f"{qid}.npz")
        try: 
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
        except:
            q_feat = np.zeros((10, self.q_feat_dim)).astype(np.float32)
            logger.info(f"Something wrong when loading the query feature {q_feat_path}.")

        if self.q_feat_type == "last_hidden_state":
            # q_feat = q_feat[:self.max_q_l]
            q_feat = q_feat
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, meta):
        dset_name = meta['dset_name']
        v_feat_suffix = meta['v_feat_suffix']
        vid = meta['vid']

        v_feat_list = []
        for feat_type, _feat_dir in zip(self.v_feat_types, self.v_feat_dirs):
            v_feat_dir = _feat_dir + v_feat_suffix
            if self.use_cache > 0:
                _feat = self.vid_cache[feat_type][vid]
            else:
                _feat_path = os.path.join('data', dset_name, v_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"].astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

class DatasetMR(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """
    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir, v_feat_dim, q_feat_dim,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 use_cache=-1, fix_len=-1, add_easy_negative=1, easy_negative_only=-1):
        self.dset_name = dset_name
        self.data_path = data_path[0] if isinstance(data_path, list) else data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.v_feat_dim = v_feat_dim
        self.q_feat_dim = q_feat_dim
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.fix_len = fix_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.use_cache = use_cache
        self.add_easy_negative = add_easy_negative
        self.easy_negative_only = easy_negative_only
        
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

        self.v_feat_types = [feat_dir.split('/')[-1] for feat_dir in self.v_feat_dirs]
        t_feat_type = q_feat_dir.split('/')[-1]

        if self.use_cache > 0:
            print('Loading the off-line features...')
            dset_dir = os.path.join('data', self.dset_name)
            vid_keys = [meta['vid'] for meta in self.data]
            qid_keys = [meta['qid'] for meta in self.data]

            self.vid_cache = {}
            for v_feat_type in self.v_feat_types:
                assert 'vid' in v_feat_type
                with h5py.File(os.path.join(dset_dir, 'h5py', v_feat_type + '.hdf5'), 'r') as f:
                    self.vid_cache[v_feat_type] = {key: f[str(key)][:] for key in tqdm(vid_keys)}

            assert 'txt' in t_feat_type
            self.txt_cache = {}
            with h5py.File(os.path.join(dset_dir, 'h5py', t_feat_type + '.hdf5'), 'r') as f:
                for key in tqdm(qid_keys):
                    try:
                        self.txt_cache[key] = f[str(key)][:]
                    except:
                        logger.info(f"text {key} is not in the cache.")

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)

        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.dset_name in ['hacs', 'ego4d', 'videocc', 'activitynet']:
            for i, window_i in enumerate(meta["relevant_windows"]):
                if window_i[1] - window_i[0] < self.clip_len:
                    center = (window_i[1] + window_i[0]) / 2 
                    window_i[0] = max(0, center - 0.5 * self.clip_len)
                    window_i[1] = min(float(meta['duration']), center + 0.5 * self.clip_len)
                    window_i[1] = max(self.clip_len, window_i[1])

        model_inputs["timestamp"] = ( (torch.arange(0, ctx_l) + self.clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

        if 'test' in self.data_path and 'qvhighlights' in self.dset_name:
            meta["relevant_windows"] = [[0, 150]]
        relevant_windows = torch.Tensor(meta["relevant_windows"])

        # assign the nearest window for each timestamp i.e., qvhighlights.
        num_vid_seq = model_inputs["timestamp"].shape[0]
        num_windows = relevant_windows.shape[0]

        relevant_windows_ts = relevant_windows / (ctx_l * self.clip_len)
        relevant_windows_ts = relevant_windows_ts.unsqueeze(0).repeat(num_vid_seq, 1, 1)
        model_inputs_ts = model_inputs["timestamp"].unsqueeze(1).repeat(1, num_windows, 1)

        if meta['qid'] is not None:
            nn_window_ts = torch.zeros_like(model_inputs["timestamp"])
            diff_left = model_inputs_ts[..., 0]  - relevant_windows_ts[..., 0]
            diff_right = relevant_windows_ts[..., 1] - model_inputs_ts[..., 1]
            assign_idx = torch.where((diff_left >= 0) * (diff_right >= 0))
            if min(assign_idx[0].shape) == 0:   # not assigned, happened in activitynet.
                nn_window_ts = relevant_windows_ts.squeeze(1)
            else:
                nn_window_ts[assign_idx[0]] = relevant_windows_ts[assign_idx[0], assign_idx[1]]

        model_inputs["span_labels_nn"] = nn_window_ts
        model_inputs["timestamp_window"] = 1 * (model_inputs["timestamp"][:,0] >= nn_window_ts[:,0])  & (model_inputs["timestamp"][:,1] <= nn_window_ts[:,1])

        # for activitynet.
        if model_inputs["timestamp_window"].sum() < 1:
            idx = int(meta['relevant_windows'][0][0] / self.clip_len)
            idx = max(0, min(idx, ctx_l-1))
            model_inputs["timestamp_window"][idx] = 1

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
            if 'saliency_scores' in meta.keys():
                model_inputs["saliency_scores"] = torch.zeros(ctx_l).double()
                limit = meta["relevant_clip_ids"].index(ctx_l) if (np.array(meta["relevant_clip_ids"]) >= ctx_l).any() else None
                model_inputs["saliency_scores"][meta["relevant_clip_ids"][:limit]] = torch.tensor(np.mean(np.array(meta["saliency_scores"][:limit]), -1))
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
            else:
                model_inputs["saliency_scores"] = model_inputs["timestamp_window"]
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)  # only one gt
                model_inputs["saliency_pos_labels"] = [ random.choice(torch.where(model_inputs['saliency_scores'])[0].tolist()) ]

        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=1):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_st = min(gt_st, ctx_l-1)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_ed = gt_st

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st] * max_n #[gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))

        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]

        if agg_scores[sort_indices[-1]] == agg_scores[sort_indices[0]]:
            hard_neg_clip_indices = hard_pos_clip_indices

        easy_pos_clip_indices = []
        easy_neg_clip_indices = []

        if self.add_easy_negative > 0:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        if self.easy_negative_only > 0:
            return easy_pos_clip_indices, easy_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        if self.use_cache > 0:
            try:
                q_feat = self.txt_cache[qid]
            except:
                q_feat = np.zeros((10, self.q_feat_dim)).astype(np.float32)
            return  torch.from_numpy(q_feat)

        q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
        try: 
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
        except:
            q_feat = np.zeros((10, self.q_feat_dim)).astype(np.float32)
            logger.info(f"Something wrong when loading the query feature {q_feat_path}.")

        if self.q_feat_type == "last_hidden_state":
            # q_feat = q_feat[:self.max_q_l]
            q_feat = q_feat
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for feat_type, _feat_dir in zip(self.v_feat_types, self.v_feat_dirs):
            if self.use_cache > 0:
                _feat = self.vid_cache[feat_type][vid]
            else:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"].astype(np.float32)
                # _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

class DatasetHL(Dataset):
    def __init__(self,
                 dset_name,
                 domain,
                 data_path,
                 v_feat_types, 
                 v_feat_dirs, 
                 t_feat_dir,
                 use_tef=False
                 ):
        assert dset_name in ['tvsum', 'youtube']
        self.dset_name = dset_name
        dset_domain = {'tvsum': TVSUM_SPLITS,
                       'youtube': YOUTUBE_SPLITS}
        self.splits = dset_domain[dset_name]
        assert domain in self.splits.keys()

        self.domain = domain
        assert len(data_path) == 1
        self.data_path = data_path[0] if isinstance(data_path, list) else data_path
        self.v_feat_types = v_feat_types.split('_')
        self.v_feat_dirs = v_feat_dirs
        self.q_feat_type = "last_hidden_state"
        self.q_feat_dir = t_feat_dir

        self.txt_drop_ratio = 0
        self.normalize_t = True
        self.normalize_v = True

        self.label = nncore.load(self.data_path)
        self.use_tef = use_tef

        self.video_id = {
            k: [s for s in self.splits[domain][k] if s in self.label]
            for k in ('train', 'val')
        }
        self.set_state('train')

    def __len__(self):
        return len(self.video_id[self.state])

    def __getitem__(self, idx):
        vid = self.get_video_id(idx)
        video = self._get_video_feat_by_vid(vid)
        saliency = self.get_saliency(idx)

        if self.dset_name == 'youtube':
            saliency_pos_labels = torch.Tensor([random.choice(torch.where(saliency > 0)[0].tolist())])
        elif self.dset_name == 'tvsum':
            saliency_pos_labels = torch.Tensor([random.choice(torch.where(saliency > 0)[0].tolist())])
            # saliency_pos_labels = torch.Tensor([random.choice(torch.where(saliency != min(saliency))[0].tolist())])
        else:
            raise NotImplementedError

        num_clips = min(c.size(0) for c in (video, saliency))

        video = video[:num_clips]
        saliency = saliency[:num_clips]

        if self.use_tef:
            ctx_l = video.shape[0]
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            video = torch.cat([video, tef], dim=1)  # (Lv, Dv+2)

        data = dict(
            video=DataContainer(video),
            saliency=DataContainer(saliency, pad_value=-1),
            saliency_pos_labels=saliency_pos_labels)

        if self.q_feat_dir is not None:
            query = self._get_query_feat_by_qid(vid)
            data['query'] = DataContainer(query, pad_value=float('inf'))
        return data

    def set_state(self, state):
        self.state = 'train' if state == 'train' else 'val'

    def get_video_id(self, idx):
        return self.video_id[self.state][idx]

    def get_video(self, idx):
        video_id = self.get_video_id(idx)
        video = torch.from_numpy(self.video[video_id]).float()
        optic = torch.from_numpy(self.optic[video_id]).float()
        return torch.cat((video, optic), dim=1)

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for feat_type, _feat_dir in zip(self.v_feat_types, self.v_feat_dirs):
            # if self.use_cache > 0:
                # _feat = self.vid_cache[feat_type][vid]
            # else:
            if True:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"].astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

    def _get_query_feat_by_qid(self, qid):
        # if self.use_cache > 0:
        #     try:
        #         q_feat = self.txt_cache[qid]
        #     except:
        #         q_feat = np.zeros((10, self.q_feat_dim)).astype(np.float32)
        #     return  torch.from_numpy(q_feat)

        q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
        try: 
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
        except:
            q_feat = np.zeros((10, self.q_feat_dim)).astype(np.float32)
            logger.info(f"Something wrong when loading the query feature {q_feat_path}.")

        if self.q_feat_type == "last_hidden_state":
            # q_feat = q_feat[:self.max_q_l]
            q_feat = q_feat
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def get_saliency(self, idx):
        if self.dset_name == 'tvsum':
            video_id = self.get_video_id(idx)
            saliency = torch.Tensor(self.label[video_id]['anno'])

            # top-5 saliency scores as a threshold.
            # saliency_tmp = saliency.mean(1)
            # topk = int(saliency_tmp.shape[0] * 0.1)
            # th = saliency_tmp[torch.sort(saliency_tmp)[1][-topk]] # v4
            # saliency = saliency_tmp - th
    
            # saliency_tmp = saliency.mean(1) # med
            # th = saliency_tmp.median()
            # saliency = saliency_tmp - th

            saliency = (saliency - saliency.mean()).mean(dim=1)
            # saliency = (saliency.sum(dim=1) - 20) / 80  # v2

        elif self.dset_name == 'youtube':
            video_id = self.get_video_id(idx)
            saliency = [1 if s > 0 else 0 for s in self.label[video_id]['match']]
        else:
            raise NotImplementedError
        return torch.Tensor(saliency)

    def evaluate(self, blob, k=5, save_dir=None, **kwargs):
        # blob = nncore.to_dict_of_list(blob)
        collected = []
        
        if save_dir is not None:
            import json
            with open(os.path.join(save_dir, self.dset_name, self.domain +'.jsonl'), 'w') as f:
                for idx, score in enumerate(blob):
                    video_id = self.get_video_id(idx)
                    entry = {'vid':video_id, 'pred': score[0].tolist(), 'gt': self.get_saliency(idx).tolist(), 
                                        'duration': int(self.label[video_id]['frames']) / int(self.label[video_id]['fps']),
                                        'domain': self.label[video_id]['domain'], 'fps': self.label[video_id]['fps']}
                    if self.dset_name == 'tvsum':
                        entry.update({'title':self.label[video_id]['title']}) 
                    if self.dset_name == 'youtube':
                        entry.update({'clip':self.label[video_id]['clip']})
                    f.write(json.dumps(entry) + '\n')

        if self.dset_name == 'tvsum':
            for i in range(20):
                video_ap = []
                for idx, score in enumerate(blob):
                    inds = torch.argsort(score[0], descending=True)
                    video_id = self.get_video_id(idx)
                    label = torch.Tensor(self.label[video_id]['anno'])[:, i]
                    label = torch.where(label > label.median(), 1.0, .0)
                    label = label[inds].tolist()[:k]

                    if (num_gt := sum(label)) == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(label):
                        hits += gt
                        _rec = hits / num_gt
                        _prc = hits / (j + 1)
                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc
                    video_ap.append(ap)
                collected.append(sum(video_ap) / len(video_ap))

        elif self.dset_name == 'youtube':
            for idx, score in enumerate(blob):
                inds = torch.argsort(score[0], descending=True)
                label = self.get_saliency(idx)[inds].tolist()

                if (num_gt := sum(label)) == 0:
                    collected.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for i, gt in enumerate(label):
                    hits += gt
                    _rec = hits / num_gt
                    _prc = hits / (i + 1)
                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc
                collected.append(ap)
        else:
            raise NotImplementedError

        mean_ap = sum(collected) / len(collected)
        results = dict(mAP=round(mean_ap, 5))
        return results

class DatasetQFVS(Dataset):
    def __init__(self,config, use_tef=True):
        # pdb.set_trace()
        self.config=config
        self.dataset=[]
        self.use_tef=use_tef

        self.embedding=load_pickle(f"./data/qfvs/txt_clip/{self.config['txt_feature']}.pkl")
 
        for video_id in self.config["train_videos"]:
            for _ , _, files in os.walk("./data/qfvs/metadata/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
                for file in files:
                    self.dataset.append(file[:file.find("_oracle.txt")]+"_"+str(video_id))

    def __getitem__(self,index):
        video_id=self.dataset[index].split('_')[2]
        feat_type = self.config['vid_feature']
        # pdb.set_trace()
        feat_type = self.config['vid_feature']
        f=h5py.File(f'./data/qfvs/processed/P0{video_id}_{feat_type}.h5','r')
        features=f['feature'][()]
        # dim=features.shape[-1]
        # features=features.reshape(-1, dim)
        # seg_len=f['seg_len'][()]
        dim = features.shape[-1]
        ctx_l = features.shape[0]
        seg_len = np.ones(ctx_l)

        # mask = torch.zeros(self.config["max_segment_num"], self.config["max_frame_num"], dtype=torch.bool)
        # for j in range(len(seg_len)):
            # for k in range(seg_len[j]):
                # mask[j][k] = 1

        # ctx_l = seg_len.sum()
        features = torch.from_numpy(features)
        # features = features[mask, :]

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            features = torch.cat([features, tef], dim=1)  # (Lv, Dv+2)

        transfer={"Cupglass":"Glass",
                  "Musicalinstrument":"Instrument",
                  "Petsanimal":"Animal"}

        concept1,concept2=self.dataset[index].split('_')[0:2]

        concept1_GT=torch.zeros(ctx_l)
        concept2_GT=torch.zeros(ctx_l)
        with open("./data/qfvs/metadata/origin_data/Dense_per_shot_tags/P0"+video_id+"/P0"+video_id+".txt","r") as f:
            lines=f.readlines()
            for index,line in enumerate(lines):
                concepts=line.strip().split(',')
                if concept1 in concepts:
                    concept1_GT[index]=1
                if concept2 in concepts:
                    concept2_GT[index]=1

        # shot_num=seg_len.sum()
        # mask_GT=torch.zeros(ctx_l)
        # for i in range(shot_num):
            # mask_GT[i]=1
        mask_GT=torch.ones(ctx_l)

        oracle_summary = torch.zeros(ctx_l)
        GT_summary_shots = []
        with open("./data/qfvs/metadata/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)+"/"+str(concept1)+"_"+str(concept2)+"_"+"oracle.txt","r") as f:
            for line in f.readlines():
                GT_summary_shots.append(int(line.strip()))
        GT_summary_shots = [x - 1 for x in GT_summary_shots]
        for element in GT_summary_shots:
            oracle_summary[element] = 1

        if concept1 in transfer:
            concept1=transfer[concept1]
        if concept2 in transfer:
            concept2=transfer[concept2]
        concept1=self.embedding[concept1]
        concept2=self.embedding[concept2]

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

def start_end_collate_mr(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue

        batched_data[k] = pad_sequences_1d(
        [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data

def start_end_collate_hl(batch):
    model_inputs_keys = batch[0].keys()

    batched_data = dict()
    for k in model_inputs_keys:
        batched_data[k] = pad_sequences_1d([e[k].data for e in batch], dtype=torch.float32, fixed_length=None)
    return batched_data

def start_end_collate_qfvs(batch):
    model_inputs_keys = batch[0].keys()

    batched_data = dict()
    for k in model_inputs_keys:
        batched_data[k] = pad_sequences_1d([e[k].data for e in batch], dtype=torch.float32, fixed_length=None)

    return batched_data

def prepare_batch_inputs_mr(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    targets = {}
    targets['timestamp'] = batched_model_inputs["timestamp"][0].to(device, non_blocking=non_blocking)
    targets['timestamp_mask'] = batched_model_inputs["timestamp"][1].to(device, non_blocking=non_blocking)
    targets['timestamp_window'] = batched_model_inputs["timestamp_window"][0].to(device, non_blocking=non_blocking)
    targets['span_labels_nn'] = batched_model_inputs["span_labels_nn"][0].to(device, non_blocking=non_blocking)

    if 'saliency_scores' in batched_model_inputs.keys():
        targets['saliency_scores'] = batched_model_inputs["saliency_scores"][0].to(device, non_blocking=non_blocking)

    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "weight_ablation" in batched_model_inputs:
        targets["weight_ablation"] = batched_model_inputs["weight_ablation"][0].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets

def prepare_batch_inputs_hl(batched_model_inputs, device='cuda', non_blocking=False):
    src_vid = batched_model_inputs['video'][0].to(device, non_blocking=non_blocking)
    src_vid_mask = batched_model_inputs['video'][1].bool().to(device, non_blocking=non_blocking)
    src_txt = batched_model_inputs['query'][0].to(device, non_blocking=non_blocking) \
        if 'query' in batched_model_inputs.keys() else None
    src_txt_mask = batched_model_inputs['query'][1].bool().to(device, non_blocking=non_blocking) \
        if 'query' in batched_model_inputs.keys() else None

    model_inputs = dict(
        src_vid=src_vid, src_vid_mask=src_vid_mask,
        src_txt=src_txt, src_txt_mask=src_txt_mask)

    # if 'audio' in batched_model_inputs.keys():
    #     src_aud = batched_model_inputs['audio'][0].bool().to(device, non_blocking=non_blocking)
    #     src_aud_mask = batched_model_inputs['audio'][1].bool().to(device, non_blocking=non_blocking)
    #     model_inputs['src_aud']=src_aud;  model_inputs['src_aud_mask']=src_aud_mask;

    targets = {}
    saliency = batched_model_inputs['saliency'][0].to(device, non_blocking=non_blocking)
    saliency_pos_labels = batched_model_inputs['saliency_pos_labels'][0].to(device, non_blocking=non_blocking)

    targets['saliency_scores'] = saliency
    targets['saliency_pos_labels'] = saliency_pos_labels.long()
    targets['timestamp_mask'] = batched_model_inputs["video"][1].to(device, non_blocking=non_blocking)
    targets['timestamp_window'] = 1 * (saliency > 0)

    return model_inputs, targets

def prepare_batch_inputs_qfvs(data, config, eval=False):
    if not eval:
        features, mask, seg_len, \
        concept1_GT, concept2_GT, mask_GT, oracle_summary_GT, \
        src_txt_1, src_txt_2, src_txt_mask_1, src_txt_mask_2,\
        saliency_pos_labels_1, saliency_pos_labels_2, saliency_pos_labels_oracle = \
            data['features'][0], data['features'][1], data['seg_len'][0],\
            data['concept1_GT'][0], data['concept2_GT'][0], data['mask_GT'][0], data['oracle_summary'][0],\
            data['tokens_pad1'][0], data['tokens_pad2'][0], data['tokens_pad1'][1], data['tokens_pad2'][1], \
            data['saliency_pos_labels_1'][0], data['saliency_pos_labels_2'][0], data['saliency_pos_labels_oracle'][0],
    else:
        features, mask, seg_len, \
        src_txt_1, src_txt_2, src_txt_mask_1, src_txt_mask_2 =  \
            data['features'][0], data['features'][1], data['seg_len'][0],\
            data['tokens_pad1'][0], data['tokens_pad2'][0], data['tokens_pad1'][1], data['tokens_pad2'][1]

    # preprocess for vid input.
    seq = features.to('cuda')
    mask = mask.to('cuda')

    # for txt input.
    src_txt_1 = src_txt_1.to(torch.float32).to('cuda')
    src_txt_2 = src_txt_2.to(torch.float32).to('cuda')
    src_txt_mask_1 = src_txt_mask_1.to('cuda')
    src_txt_mask_2 = src_txt_mask_2.to('cuda')

    src_txt_oracle = torch.cat((src_txt_1, src_txt_2), dim=1).to('cuda')
    src_txt_mask_oracle = torch.cat((src_txt_mask_1, src_txt_mask_2), dim=1).to('cuda')

    model_inputs_1 = dict(src_vid=seq, src_vid_mask=mask, src_txt=src_txt_1, src_txt_mask=src_txt_mask_1)
    model_inputs_2 = dict(src_vid=seq, src_vid_mask=mask, src_txt=src_txt_2, src_txt_mask=src_txt_mask_2)
    model_inputs_oracle = dict(src_vid=seq, src_vid_mask=mask, src_txt=src_txt_oracle, src_txt_mask=src_txt_mask_oracle)

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
        return model_inputs_1, model_inputs_2, model_inputs_oracle, mask
