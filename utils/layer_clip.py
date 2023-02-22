# This function is use to layer clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

def clip_to_shot(clip_gt_id, clip_shot_id):
    '''
    clip_input_id = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    clip_gt_id = [1,3,0,7,12,9,11,2,10,5,4,6,8]
    clip_shot_id = [283,284,283,285,287,286,287,284,286,285,284,285,286]
    ->
    clip_input_id = [0,2,1,7,10,3,9,11,5,8,12,4,6]
    clip_gt_id = [1,3,0,7,12,9,11,2,10,5,4,6,8]
    clip_shot_id = [283,284,283,285,287,286,287,284,286,285,284,285,286]    
    ->
    clip_gt_id = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    clip_shot_id = [283,283,284,284,284,285,285,285,286,286,286,287,287]

    output:
    [[0, 2], [1, 7, 10], [3, 9, 11], [5, 8, 12], [4, 6]]
    [[1, 0], [3, 2, 4], [7, 5, 6], [9, 10, 8], [12, 11]]
    '''
    clip_input_id = [i for i in range(len(clip_gt_id))]
    if torch.is_tensor(clip_gt_id): clip_gt_id = clip_gt_id.cpu().numpy().tolist()
    if torch.is_tensor(clip_shot_id): clip_shot_id = clip_shot_id.cpu().numpy().tolist()

    zip_list = zip(clip_input_id, clip_gt_id, clip_shot_id)
    zip_list_sorted = sorted(zip_list, key=lambda x:(x[2]))

    clip_input_id_sorted, clip_gt_id_sorted, clip_shot_id_sorted = zip(*zip_list_sorted)

    clip_input_id_sorted = list(clip_input_id_sorted)
    clip_gt_id_sorted = list(clip_gt_id_sorted)
    clip_shot_id_sorted = list(clip_shot_id_sorted)

    shot_len_list = [i[1] for i in Counter(clip_shot_id_sorted).items()]
    
    idx = 0
    shot_input_id = []
    for shot_len in shot_len_list:
        shot_input_id_ele = []
        for i in range(shot_len):
            shot_input_id_ele.append(clip_input_id_sorted[idx])
            idx += 1
        shot_input_id.append(shot_input_id_ele)

    idx = 0
    shot_gt_id = []
    for shot_len in shot_len_list:
        shot_gt_id_ele = []
        for i in range(shot_len):
            shot_gt_id_ele.append(clip_gt_id_sorted[idx])
            idx += 1
        shot_gt_id.append(shot_gt_id_ele)
    
    return shot_input_id, shot_gt_id

def clip_to_scene(clip_gt_id, clip_shot_id, clip_scene_id):
    '''
    clip_input_id = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    clip_gt_id = [1,10,9,5,11,2,3,7,8,4,0,12,6]
    clip_shot_id = [6,10,9,8,10,6,6,8,9,7,5,11,8]
    clip_scene_id = [4,6,5,5,6,4,4,5,5,4,3,6,5]
    ->
    shot_input_id = [[10], [0, 5, 6], [9], [3, 7, 12], [2, 8], [1, 4], [11]]
    shot_gt_id = [[0], [1, 2, 3], [4], [5, 7, 6], [9, 8], [10, 11], [12]]
    ->
    scene_input_id = [[[10]], [[0, 5, 6], [9]], [[3, 7, 12], [2, 8]], [[1, 4], [11]]]
    scene_gt_id = [[[0]], [[1, 2, 3], [4]], [[5, 7, 6], [9, 8]], [[10, 11], [12]]]
    '''
    if torch.is_tensor(clip_scene_id): clip_scene_id = clip_scene_id.cpu().numpy().tolist()
    shot_input_id, shot_gt_id = clip_to_shot(clip_gt_id, clip_shot_id)

    scene_len_list = [i[1] for i in Counter(sorted(clip_scene_id)).items()]

    idx = 0
    scene_input_id = []
    scene_gt_id = []
    for scene_len in scene_len_list:
        scene_input_id_ele = []
        scene_gt_id_ele = []
 
        jdx = 0
        while jdx < scene_len:
            scene_input_id_ele.append(shot_input_id[idx])
            scene_gt_id_ele.append(shot_gt_id[idx])
            jdx += len(shot_input_id[idx])
            idx += 1
        scene_input_id.append(scene_input_id_ele)
        scene_gt_id.append(scene_gt_id_ele)

    return scene_input_id, scene_gt_id

if __name__ == '__main__':
    clip_gt_id = torch.tensor([1,10,9,5,11,2,3,7,8,4,0,12,6])
    clip_shot_id = torch.tensor([6,10,9,8,10,6,6,8,9,7,5,11,8])
    clip_scene_id = torch.tensor([4,6,5,5,6,4,4,5,5,4,3,6,5])

    print(clip_to_scene(clip_gt_id, clip_shot_id, clip_scene_id))