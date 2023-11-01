import os
import sys
import pdb
import json
import h5py
import numpy as np
import torch
from tqdm import tqdm

# pre-package all features as h5py format, and avoid any I/O reading during training.

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def l2_normalize_np_array(np_array, eps=1e-5):
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

def write_hdf5(outfile, arr_dict):
	with h5py.File(outfile, 'w') as f:
		for key in arr_dict.keys():
			f.create_dataset(key, data=arr_dict[key])

def load_data(data_path):
	datalist = load_jsonl(data_path)
	return datalist

def _get_video_feat_by_vid(_feat_dir, vid):
	_feat_path = os.path.join(_feat_dir, f"{vid}.npz")
	_feat = np.load(_feat_path)["features"].astype(np.float32)
	_feat = l2_normalize_np_array(_feat)
	# return torch.from_numpy(v_feat)  # (Lv, D)
	return _feat  # (Lv, D)

def _get_query_feat_by_qid(q_feat_dir, qid):
	q_feat_path = os.path.join(q_feat_dir, f"{qid}.npz")
	q_feat = np.load(q_feat_path)["last_hidden_state"].astype(np.float32)
	q_feat = l2_normalize_np_array(q_feat)
	# return torch.from_numpy(q_feat)
	return q_feat

if __name__ == "__main__":
	dir_path = "/data/home/qinghonglin/univtg/data"
	dset = sys.argv[1] #"qvhighlights"
	feat_type = sys.argv[2] #"vid_slowfast"

	feat_path = os.path.join(dir_path, dset, feat_type)
	file_dir = os.listdir(feat_path)

	cache = {}
	for file in tqdm(file_dir):
		if not file.endswith('.npz'):
			continue

		file = os.path.splitext(file)[0]
		# if file.startswith('qid'):		# for qv hl
			# file = file[3:]

		if 'txt' in feat_type:
			try:
				cache[file] = _get_query_feat_by_qid(feat_path, file)
			except:
				print(f"{feat_path}/{file} is not existed.")
		elif 'vid' in feat_type:
			cache[file] = _get_video_feat_by_vid(feat_path, file)

	save_dir = os.path.join(dir_path, dset, 'h5py')
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)

	save_path = os.path.join(save_dir, feat_type + '.hdf5')
	write_hdf5(save_path , cache)

"""
python3 create_h5py.py qvhighlights txt_clip
python3 create_h5py.py video2gif vid_clip \
python3 create_h5py.py video2gif vid_slowfast \
python3 create_h5py.py video2gif txt_clip
python3 create_h5py.py charades vid_clip \
python3 create_h5py.py charades vid_slowfast \
python3 create_h5py.py charades txt_clip

python3 create_h5py.py ego4d vid_clip
python3 create_h5py.py ego4d vid_slowfast

python3 create_h5py.py tacos vid_clip; python3 create_h5py.py tacos vid_slowfast; python3 create_h5py.py tacos txt_clip; python3 create_h5py.py ego4d txt_clip;

python3 create_h5py.py videocc txt_clip_concept

"""
