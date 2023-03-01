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
data_path = '/home/jianghui/dataset/VideoReorder-MovieNet'
split = 'val'
val_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='')
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=lambda x: x)

# loss
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)

# scene order on clip model
scene_model = OneLayer()
checkpoint = torch.load(Path('./checkpoint', f'scene_to_clip_best_2023-03-01.pth'))
scene_model.to(device)
scene_model.eval()

# scene embed on clip model
scene_img_embed = scene_model.img_embedding
scene_text_embed = scene_model.text_embedding

# shot order on scene model
shot_model = OneLayer()
checkpoint = torch.load(Path('./checkpoint', f'shot_to_scene_best_2023-03-01.pth'))
shot_model.to(device)
shot_model.eval()

# frame order on shot model
frame_model = OneLayer()
checkpoint = torch.load(Path('./checkpoint', f'frame_to_shot_best_2023-03-01.pth'))
frame_model.to(device)
frame_model.eval()

DEBUG = True
score_dict = {
    'init':[],
    'scene_cluster' : [],
    'scene_order': [],
}

for data in tqdm(val_dataloader):
    # load data
    img_features, text_features, gt_id, shot_id, scene_id = data[0]
    # all_features = [img_features[i][:,0,:].reshape(-1) for i in range(len(gt_id))]
    # all_features = [torch.cat((img_features[i][:,0,:].reshape(-1), text_features[i][:,0,:].reshape(-1)), dim=0) for i in range(len(gt_id))]
    all_features = [torch.cat((scene_img_embed(img_features[i].to(device)).mean(1).reshape(-1), scene_text_embed(text_features[i].to(device)).mean(1).reshape(-1)), dim=0).cpu() for i in range(len(gt_id))]
    input_id = [i for i in range(len(gt_id))]

    # begin
    pred = list_to_one_dim(input_id)
    gt = list_to_one_dim(gt_id)
    assert(len(pred) == len(gt_id))
    init_score = DoubleLengthMatching(pred, gt)
    score_dict['init'].append(init_score)

    # scene cluster
    gt_scene_clustered = frame2scene(gt_id, scene_id)
    input_id = KMeanCLustering(features=all_features, input_id=input_id, gt_clusters=gt_scene_clustered,layer='scene')

    pred = list_to_one_dim(input_id)
    gt = list_to_one_dim(gt_id)
    assert(len(pred) == len(gt_id))
    scene_cluster_score = DoubleLengthMatching(pred, gt)
    score_dict['scene_cluster'].append(scene_cluster_score)

    # scene order
    N_scene = len(input_id)
    img_features_scene = []
    text_features_scene = []
    for idx, input_id_ele in enumerate(input_id):
        img_features_scene.append(torch.cat([img_features[i] for i in input_id_ele], dim=1))
        text_features_scene.append(torch.cat([text_features[i] for i in input_id_ele], dim=1))

    score_square = [[float('-inf') for i in range(N_scene)]for i in range(N_scene)]
    for I in range(N_scene):
        for J in range(N_scene):
            if I == J: continue
            output = scene_model([[img_features_scene[I].to(device), img_features_scene[J].to(device)], [text_features_scene[I].to(device), text_features_scene[J].to(device)]]).squeeze(0).to(device)
            score_square[I][J] = torch_to_list(output[1]-output[0])
    
    scene_order = beam_search_all(score_square)['path']

    input_id = same_shuffle(input_id, scene_order)
    img_features_scene = same_shuffle(img_features_scene, scene_order)
    text_features_scene = same_shuffle(text_features_scene, scene_order)

    pred = list_to_one_dim(input_id)
    gt = list_to_one_dim(gt_id)
    assert(len(pred) == len(gt_id))
    scene_order_score = DoubleLengthMatching(pred, gt)
    score_dict['scene_order'].append(scene_order_score)    

    continue

    # shot cluster
    gt_shot_clustered = frame2all(gt_id, shot_id, scene_id)
    features, input_id = KMeanCLustering(features=features, input_id=input_id, gt_clusters=gt_shot_clustered,layer='shot')

    pred = list_to_one_dim(input_id)
    assert(len(pred) == len(gt_id))

    N_scene = len(features)
    idx = 0
    while idx < N_scene:
        N_shot = len(input_id[idx])
        jdx = 0
        while jdx < N_shot:
            if input_id[idx][jdx] == []:
                input_id[idx].pop(jdx)
                features[idx].pop(jdx)
                jdx -= 1
                N_shot -= 1
            jdx += 1
        if input_id[idx] in [[], [[]]]:
            input_id.pop(idx)
            features.pop(idx)
        idx += 1

    # shot order 
    N_scene = len(features)
    for idx in range(N_scene):
        N_shot = len(input_id[idx])
        try:
            features_shot = [torch.mean(torch.stack(i, dim=0), dim=0) for i in features[idx]]
        except:
            print(input_id)
            print(gt_shot_clustered)
            assert False

        score_square = [[float('-inf') for i in range(N_shot)]for i in range(N_shot)]

        for I in range(N_shot):
            for J in range(N_shot):
                if I == J: continue
                output = shot_model(torch.concat((features_shot[I], features_shot[J])).unsqueeze(0).to(device))
                score_square[I][J] = torch_to_list(output[0][1]-output[0][0])   

        shot_order = beam_search_all(score_square)['path']

        features[idx] = same_shuffle(features[idx], shot_order)
        input_id[idx] = same_shuffle(input_id[idx], shot_order)

    pred = list_to_one_dim(input_id)
    assert(len(pred) == len(gt_id))

    # frame order
    N_scene = len(input_id)
    for idx in range(N_scene):
        N_shot = len(input_id[idx])
        for jdx in range(N_shot):
            N_frame = len(input_id[idx][jdx])
            features_frame = features[idx][jdx]
            score_square = [[float('-inf') for i in range(N_frame)]for i in range(N_frame)]

            for I in range(N_frame):
                for J in range(N_frame):
                    if I == J: continue
                    output = frame_model(torch.concat((features_frame[I], features_frame[J])).unsqueeze(0).to(device))
                    score_square[I][J] = torch_to_list(output[0][1]-output[0][0])

            frame_order = beam_search_all(score_square)['path']

            features[idx][jdx] = same_shuffle(features[idx][jdx], frame_order)
            input_id[idx][jdx] = same_shuffle(input_id[idx][jdx], frame_order)

    pred = list_to_one_dim(input_id)
    assert(len(pred) == len(gt_id))

    score = DoubleLengthMatching(pred, gt_id)
    score_dict['total_score'].append(score)

score_dict['init'] = sum(score_dict['init']) / len(score_dict['init'])
score_dict['scene_cluster'] = sum(score_dict['scene_cluster']) / len(score_dict['scene_cluster'])
score_dict['scene_order'] = sum(score_dict['scene_order']) / len(score_dict['scene_order'])


print(score_dict)