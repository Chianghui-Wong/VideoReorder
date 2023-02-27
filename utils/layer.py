# This function is use to layer clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from .tools import *

'''
all: have both shot and scene information on gt list
'''

def frame2shot(gt_id, shot_id):
    '''
    example:
    Input:
        gt_id = [10, 0, 5, 6, 9, 3, 12, 7, 8, 2, 1, 4, 11]
        shot_id = [6, 10, 9, 8, 10, 6, 6, 8, 9, 7, 5, 11 ,8]
    Output:
        [[10], [0, 5, 6], [9], [3, 12, 7], [8, 2], [1, 4], [11]]
    '''

    return group_by_class(gt_id, sorted(class_start_zero(shot_id)))

def shot2all(gt_id, scene_id):
    '''
    example:
    Input:
        gt_id = [[10], [0, 5, 6], [9], [3, 12, 7], [8, 2], [1, 4], [11]]
        scene_id = [4, 6, 5, 5, 6, 4, 4, 5, 5, 4, 3, 6, 5]
    Output:
        [[[10]], [[0, 5, 6], [9]], [[3, 12, 7], [8, 2]], [[1, 4], [11]]]
    '''
    scene_label = class_start_zero(scene_id)
    scene_label = group_same_with(scene_label, gt_id)
    scene_label = [i[0] for i in scene_label]

    return frame2shot(gt_id, scene_label)

def frame2all(gt_id, shot_id, scene_id):
    '''
    example:
    Input:
        gt_id = [10, 0, 5, 6, 9, 3, 12, 7, 8, 2, 1, 4, 11]
        shot_id = [6, 10, 9, 8, 10, 6, 6, 8, 9, 7, 5, 11 ,8]
        scene_id = [4, 6, 5, 5, 6, 4, 4, 5, 5, 4, 3, 6, 5]
    Output:
        [[[10]], [[0, 5, 6], [9]], [[3, 12, 7], [8, 2]], [[1, 4], [11]]]
    '''    
    
    gt_id_scene = frame2scene(gt_id, scene_id)
    sorted_shot_id = sorted(class_start_zero(shot_id))

    idx = 0
    output = []
    for gt_id_scene_ele in gt_id_scene:
        L = len(gt_id_scene_ele)
        output.append(frame2shot(gt_id_scene_ele, sorted_shot_id[idx:idx+L]))
        idx += L
    return output


def frame2scene(gt_id, scene_id):
    '''
    example:
    Input:
        gt_id = [10, 0, 5, 6, 9, 3, 12, 7, 8, 2, 1, 4, 11]
        scene_id = [4, 6, 5, 5, 6, 4, 4, 5, 5, 4, 3, 6, 5]
    Output:
        [[10], [0, 5, 6, 9], [3, 7, 12, 2, 8], [1, 4, 11]]
    '''

    return group_by_class(gt_id, sorted(class_start_zero(scene_id)))

def all2scene(gt_id):
    '''
    example:
    Input:
        gt_id = [[[10]], [[0, 5, 6], [9]], [[3, 12, 7], [8, 2]], [[1, 4], [11]]]
    Output:
        [[10], [0, 5, 6, 9], [3, 7, 12, 2, 8], [1, 4, 11]]
    '''
    return list_del_final_dim(gt_id)

def all2shot(gt_id):

    return list_del_first_dim(gt_id)

def all2frame(gt_id):

    return list_to_one_dim(gt_id)

if __name__ == '__main__':
    clip_img_id = [1,10,9,5,11,2,3,7,8,4,0,12,6]
    clip_gt_id = [10, 0, 5, 6, 9, 3, 12,7, 8, 2, 1, 4, 11]
    clip_shot_id = [6,10,9,8,10,6,6,8,9,7,5,11,8]
    clip_scene_id = [4,6,5,5,6,4,4,5,5,4,3,6,5]

    print(all2scene([[[10]], [[0, 5, 6], [9]], [[3, 12, 7], [8, 2]], [[1, 4], [11]]]))