import pdb

import networkx as nx
from sklearn.metrics import pairwise_distances
import numpy as np
import scipy.io

def process_video_mat(video_mat):
    result = []
    for shot_vec in video_mat:
        shot_vec= shot_vec[0][0]
        result.append(shot_vec)
    result = np.array(result)
    return result


def process_mat(mat):
    videos = mat['Tags'][0]
    result = []
    for video_mat in videos:
        video_mat = video_mat[0]
        video_data = process_video_mat(video_mat)
        result.append(video_data)
    return result


def load_videos_tag(mat_path="./data/ute_query/Tags.mat"):
    mat = scipy.io.loadmat(mat_path)
    videos_tag = process_mat(mat)
    return videos_tag

def semantic_iou(a, b):
    intersection = a * b
    intersection_num = sum(intersection)
    union = a + b
    union[union>0] = 1
    union_num = sum(union)
    if union_num != 0:
        return intersection_num / union_num
    else:
        return 0


def build_graph_from_pariwise_weights(weight_matrix):
    B = nx.Graph()
    bottom_nodes = list(map(lambda x: "b-{}".format(x), list(range(weight_matrix.shape[0]))))
    top_nodes = list(map(lambda x: "t-{}".format(x), list(range(weight_matrix.shape[1]))))
    edges = []
    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            weight = weight_matrix[i][j]
            edges.append(("b-{}".format(i), "t-{}".format(j), weight))
    B.add_weighted_edges_from(edges)
    return B


def calculate_semantic_matching(machine_summary, gt_summary, video_shots_tag, video_id):
    video_shots_tag = video_shots_tag[video_id]
    machine_summary_mat = video_shots_tag[machine_summary]
    gt_summary_mat = video_shots_tag[gt_summary]
    weights = pairwise_distances(machine_summary_mat, gt_summary_mat, metric=semantic_iou)
    B = build_graph_from_pariwise_weights(weights)
    matching_edges = nx.algorithms.matching.max_weight_matching(B)
    sum_weights = 0
    i = 0
    for edge in matching_edges:
        edge_data = B.get_edge_data(edge[0], edge[1])
        sum_weights += edge_data['weight']
        i += 1
    precision = sum_weights / machine_summary_mat.shape[0]
    recall = sum_weights / gt_summary_mat.shape[0]
    f1 = 2 * precision * recall / (precision + recall)
    # return 0 / 0
    return precision, recall, f1


if __name__=='__main__':
    video_shots_tag = load_videos_tag(mat_path="./../AAAI2020/evaluation_code/Tags.mat")
    machine_summary = [1, 39, 99, 31, 778]
    gt_summary = [1, 34, 101, 29, 774]
    print(calculate_semantic_matching(machine_summary, gt_summary, video_shots_tag, video_id=0))
