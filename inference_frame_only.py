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
frame_model = OneLayer_1()
checkpoint = torch.load(Path('./checkpoint', f'frame_to_shot_best_2023-03-01.pth'))
frame_model.to(device)
frame_model.eval()

# frame order infer 
frame_infer_model = OneLayer_1_infer()
checkpoint = torch.load(Path('./checkpoint', f'frame_to_shot_best_2023-03-01.pth'))
frame_infer_model.to(device)
frame_infer_model.eval()


DEBUG = True
score_dict = {
    'init':[],
    'scene_cluster' : [],
    'scene_order': [],
}

for data in tqdm(val_dataloader):
    # load data
    img_features, text_features, gt_id, shot_id, scene_id = data[0]
    N_frame = len(gt_id)

    # frame order
    score_square = [[float('-inf') for i in range(N_frame)]for i in range(N_frame)]

    for I in range(N_frame):
        for J in range(N_frame):
            if I == J: continue
            output = frame_model([[img_features[I].to(device), img_features[J].to(device)], [text_features[I].to(device), text_features[J].to(device)]]).squeeze(0).to(device)
            score_square[I][J] = torch_to_list(output[0][1]-output[0][0])

    frame_order = beam_search_all(score_square)['path']

    img_features = same_shuffle(img_features, frame_order)
    text_features = same_shuffle(text_features, frame_order)
    input_id = same_shuffle(input_id, frame_order)

    pred = list_to_one_dim(input_id)
    assert(len(pred) == len(gt_id))

    score = DoubleLengthMatching(pred, gt_id)
    score_dict['total_score'].append(score)

score_dict['init'] = sum(score_dict['init']) / len(score_dict['init'])
score_dict['scene_cluster'] = sum(score_dict['scene_cluster']) / len(score_dict['scene_cluster'])
score_dict['scene_order'] = sum(score_dict['scene_order']) / len(score_dict['scene_order'])


print(score_dict)