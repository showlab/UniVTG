import os
import numpy as np
import pdb
import json
import torch
import tqdm

def timeconvert(s):
    return str(s//60) + ':' + str(s%60)

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def create_window(score):
    max_score = max(score)
    res = []
    valid = False
    for i, s in enumerate(score):
        if not valid and s == max_score:
            start = i * clip_len
            end = (i+1) * clip_len
            valid = True
        elif valid and s == max_score:
            end = (i+1) * clip_len
        elif valid and s != max_score:
            # pdb.set_trace()
            res.append([start, end])
            valid = False
    return res

if __name__ == "__main__":
    clip_len = 2
    topk = 5
    th = 0.05

    meta_path = os.path.join('', f'curve_{topk}_window.jsonl')
    f = open(meta_path, 'w')

    dir1 = os.listdir('./data/videocc/vid_slowfast')
    dir2 = os.listdir('./data/videocc/vid_clip')
    dir = list(set(dir1).intersection(set(dir2)))
    dir = [x for x in dir if x.endswith('.npz')]

    index = 0
    # download the oidv6-class-descriptions.json and extract their class textual feature by clip text encoder (w/ pooler)
    json_path = './teacher/oidv6-class-descriptions.json'
    save_dir = f'./data/commonsense/clip_txt/class_text.npz'

    with open(json_path,'r') as load_f:
        load_dict = json.load(load_f)

    txt_feat = torch.from_numpy(np.load(save_dir)['pooler_output']).cuda()
    for vid in tqdm.tqdm(dir):
        vid_path = os.path.join(f'./data/videocc/vid_clip', vid)
        vid_feat = torch.from_numpy(np.load(vid_path)['features']).cuda()
        if len(vid_feat) == 0:
            continue
        mm = sim_matrix(vid_feat, txt_feat)
        tmp = mm.sum(0)
        concept_idx = torch.sort(tmp, descending=True)[1][:topk]
        concept_list = [load_dict[id] for id in concept_idx]

        for i, (concept_id, concept) in enumerate(zip(concept_idx, concept_list)):
            score = mm.T[concept_id].tolist()
            score = [[s // th] for s in score]
            # score = [[round(s,2)] for s in score]
            window = create_window(score)
            sample={
            'qid': concept_id.item(), #f'{vid[:-4]}_{i}',
            'query': concept,
            'duration': float(len(vid_feat) * clip_len),
            'vid': vid[:-4],
            'relevant_clip_ids': [x for x in range(len(vid_feat))],
            # 'relevant_windows': [[0, len(vid_feat) * clip_len]],
            'relevant_windows': window,
            'saliency_scores': score
            }
            if window == []:
                continue 
            f.write(json.dumps(sample))
            f.write('\n')
f.close()
