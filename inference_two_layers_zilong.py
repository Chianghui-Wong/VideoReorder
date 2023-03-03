import numpy as np
import torch
import cv2
import json
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import os
import random
from utils import *
from models import *
import requests
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn
import time
import wandb
from scipy.special import comb, perm
import copy
import itertools
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read inference data
data_path = '/home/jax/dataset/VideoReorder-MovieNet'
split = 'val'
val_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='')
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=lambda x: x)

# loss
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)

# scene order on clip model
# scene_model = OneLayer()
# checkpoint = torch.load(Path('./checkpoint', f'scene_to_clip_best_2023-03-01.pth'))
# scene_model.load_state_dict(checkpoint,strict=False)
# scene_model.to(device)
# scene_model.eval()

# scene embed on clip model
# scene_img_embed = scene_model.img_embedding
# scene_text_embed = scene_model.text_embedding

# shot order on scene model
shot_model = OneLayer()
checkpoint = torch.load(Path('./checkpoint', f'shot_to_scene_best_2023-03-01.pth'))
shot_model.load_state_dict(checkpoint, strict=False)
shot_model.to(device)
shot_model.eval()

# frame order on shot model
frame_model = OneLayer_CL()
checkpoint = torch.load(Path('./checkpoint', f'frame_to_shot_best_2023-03-02.pth'))
frame_model.load_state_dict(checkpoint, strict=False)
frame_model.to(device)
frame_model.eval()

DEBUG = True
score_dict = {
    'init':[],
    'shot_cluster' : [],
    'shot_cluster_acc' : [],
    'shot_reorder': [],
    # 'frame_reorder': [],
    'total_score': [],
    'shot_cluster_iou' : [],
    'frame_reorder' : []
}

for data in tqdm(val_dataloader):
    # 1. load data
    img_features, text_features, gt_id, shot_id, scene_id = data[0]
    input_id = [i for i in range(len(gt_id))]
    gt_id_all = frame2all(gt_id, shot_id, scene_id)

    for gt_scene_id in  gt_id_all:
        input_scene_id = list_to_one_dim(gt_scene_id)
        random.shuffle(input_scene_id)
        img_shot_features = [img_features[i].to(device) for i in input_scene_id]
        text_shot_features = [text_features[i].to(device) for i in input_scene_id]
        gt_n_cluster = len(gt_scene_id)
        print(gt_scene_id)
        
        with torch.no_grad():
            cluster_feats = frame_model.get_cluster_feats(img_shot_features, text_shot_features)
            input_clustered_id = KMeanCLustering(cluster_feats, input_scene_id, gt_clusters=gt_n_cluster)

        # score_dict['shot_cluster_iou'].append(KMeanAcc(input_clustered_id, gt_scene_id))
        print('cluster pred', input_clustered_id)
        print('cluster gt', gt_scene_id)
        input_id.append(input_clustered_id)
    
        # shot level reorder
        img_features_shot, text_features_shot = [], []
        input_shot_reorder_id = input_clustered_id
        for input_id_ele in input_shot_reorder_id:
            if input_id_ele != []:
                img_features_shot.append(torch.cat([img_features[i] for i in input_id_ele], dim=1).to(device))
                text_features_shot.append(torch.cat([text_features[i] for i in input_id_ele], dim=1).to(device))
            else:
                img_features_shot.append(torch.zeros(1, 1, 768).to(device))
                text_features_shot.append(torch.zeros(1, 1, 512).to(device))                

        shot_len = gt_n_cluster
        score_square = [[float('-inf') for _ in range(shot_len)]for _ in range(shot_len)]
        for I in range(shot_len):
            for J in range(shot_len):
                if I == J: continue
                output = shot_model([[img_features_shot[I].to(device), img_features_shot[J].to(device)], [text_features_shot[I].to(device), text_features_shot[J].to(device)]]).squeeze(0).to(device)
                output_sfm = torch.nn.functional.softmax(output, dim=0)
                score_square[I][J] = float(output_sfm[1] - output_sfm[0])
        
        shot_order = beam_search_all(score_square)['path']
        input_shot_reorder_id = same_shuffle(input_shot_reorder_id, shot_order)

        if shot_len > 1:
            score_dict['shot_reorder'].append(DoubleLengthMatching(list_to_one_dim(input_shot_reorder_id), list_to_one_dim(gt_scene_id)))   
        print('shot reorder pred', input_shot_reorder_id)
        print('shot reorder gt', gt_scene_id)

        # frame level reorder
        input_frame_reorder_id = input_shot_reorder_id
        img_features_frame, text_features_frame = [], []
        for ele in input_shot_reorder_id:
            img_features_frame.append([img_features[i] for i in ele])
            text_features_frame.append([text_features[i] for i in ele])

        for idx, input_id_ele in enumerate(input_shot_reorder_id):
            if input_id_ele == []:
                continue
            frame_len = len(input_id_ele)
            score_square = [[float('-inf') for _ in range(frame_len)]for _ in range(frame_len)]
            # frame reorder in shot
            for I in range(frame_len):
                for J in range(frame_len):
                    if I == J: continue
                    output = frame_model.get_order_score([[img_features_frame[idx][I].to(device), img_features_frame[idx][J].to(device)], [text_features_frame[idx][I].to(device), text_features_frame[idx][J].to(device)]])
                    # output_sfm = torch.nn.functional.softmax(output, dim=0)
                    # score_square[I][J] = float(output_sfm[1] - output_sfm[0])
                    score_square[I][J] = output

            frame_order = beam_search_all(score_square)['path']
            input_frame_reorder_id[idx] = same_shuffle(input_frame_reorder_id[idx], frame_order)
        
        if frame_len > 1:
            score_dict['frame_reorder'].append(DoubleLengthMatching(list_to_one_dim(input_frame_reorder_id), list_to_one_dim(gt_scene_id)))   
        print('frame reorder pred', input_frame_reorder_id)
        print('frame reorder gt', gt_scene_id)  


    # print(sum(score_dict['shot_cluster_iou']) / len(score_dict['shot_cluster_iou']))
    
for key in score_dict:
    if len(score_dict[key]) > 0:
        score_dict[key] = sum(score_dict[key]) / len(score_dict[key])
print(score_dict)


# exit() 
#     all_features = [torch.cat((scene_img_embed(img_features[i].to(device)).mean(1).reshape(-1), scene_text_embed(text_features[i].to(device)).mean(1).reshape(-1)), dim=0).cpu() for i in range(len(gt_id))]

#     shot_num = max(shot_id) - min(shot_id) + 1
#     pred = list_to_one_dim(input_id) ; gt = list_to_one_dim(gt_id) ; assert(len(pred) == len(gt_id))
#     score_dict['init'].append(DoubleLengthMatching(pred, gt))
    
#     # 2. cluster shot
#     input_id = KMeanCLustering(all_features, input_id, gt_clusters=shot_num)
#     pred = list_to_one_dim(input_id) ; gt = list_to_one_dim(gt_id) ; assert(len(pred) == len(gt_id))
#     score_dict['shot_cluster'].append(DoubleLengthMatching(pred, gt))
#     score_dict['shot_cluster_acc'].append(KMeanAcc(input_id, frame2shot(gt, shot_id)))

#     # 2.1 make shot structure data
#     img_features_shot, text_features_shot = [], []
#     for input_id_ele in input_id:
#         img_features_shot.append(torch.cat([img_features[i] for i in input_id_ele], dim=1))
#         text_features_shot.append(torch.cat([text_features[i] for i in input_id_ele], dim=1))
    
#     # 3. shot_reorder
#     score_square = [[float('-inf') for _ in range(shot_num)]for _ in range(shot_num)]
#     for I in range(shot_num):
#         for J in range(shot_num):
#             if I == J: continue
#             output = scene_model([[img_features_shot[I].to(device), img_features_shot[J].to(device)], [text_features_shot[I].to(device), text_features_shot[J].to(device)]]).squeeze(0).to(device)
#             output_sfm = torch.nn.functional.softmax(output, dim=0)
#             score_square[I][J] = float(output_sfm[1] - output_sfm[0])
    
#     shot_order = beam_search_all(score_square)['path']
#     input_id = same_shuffle(input_id, shot_order)
#     pred = list_to_one_dim(input_id) ; gt = list_to_one_dim(gt_id) ; assert(len(pred) == len(gt_id))
#     score_dict['shot_reorder'].append(DoubleLengthMatching(pred, gt))

#     # 3.3 make frame struct data
#     img_features_frame, text_features_frame = [], []
#     for input_id_ele in input_id:
#         img_features_frame.append([img_features[i] for i in input_id_ele])
#         text_features_frame.append([text_features[i] for i in input_id_ele])

#     for idx, input_id_ele in enumerate(input_id):
#         shot_num = len(input_id_ele)
#         score_square = [[float('-inf') for _ in range(shot_num)]for _ in range(shot_num)]
#         # frame reorder in shot
#         for I in range(shot_num):
#             for J in range(shot_num):
#                 if I == J: continue
#                 value1, value2 = frame_model([[img_features_frame[idx][I].to(device), img_features_frame[idx][J].to(device)], [text_features_frame[idx][I].to(device), text_features_frame[idx][J].to(device)]])
#                 score_square[I][J] = float(value2) - float(value1)

#         frame_order = beam_search_all(score_square)['path']
#         input_id[idx] = same_shuffle(input_id[idx], frame_order)
    
#     pred = list_to_one_dim(input_id) ; gt = list_to_one_dim(gt_id) ; assert(len(pred) == len(gt_id))
#     score = DoubleLengthMatching(pred, gt)
#     score_dict['total_score'].append(score)

# for key in score_dict:
#     if len(score_dict[key]) > 0:
#         score_dict[key] = sum(score_dict[key]) / len(score_dict[key])
# print(score_dict)
