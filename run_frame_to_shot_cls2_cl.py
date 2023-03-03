import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import cv2
import json
from pathlib import Path
from PIL import Image
import os
import random
import requests
from tqdm import tqdm
import time
import wandb

from utils import *
from models import *

# from transformers import AutoProcessor, AutoTokenizer
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


timestamp = time.strftime('%Y-%m-%d', time.localtime(time.time()))
wandb.init(
    project = 'VideoReorder',
    name = 'frame to shot cls'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

data_path = '/home/jax/dataset/VideoReorder-MovieNet'
split = 'train'
train_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='frame_cl')
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=(split == 'train'), num_workers=8, pin_memory=True, collate_fn=lambda x: x)

split = 'val'
val_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='frame_cl')
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=(split == 'train'), num_workers=8, pin_memory=True, collate_fn=lambda x: x)

net = OneLayer_CL()
net.to(device)
wandb.watch(net)

lr = 1e-4
epoch = 5

# loss_func = ClipPairWiseLoss()
# loss_func.to(device)
pred_func = ClipPairWisePred()
pred_func.to(device)
optim = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.1, 0.999))


best_val_acc = 0.0
for e in range(epoch):
    print(f'epoch {e}:')

    # train
    net.train()
    loss_epoch_list = []
    score_epoch_list = []
    for batch_data in tqdm(train_dataloader):
        loss_batch_list = []
        score_batch_list = []

        loss_temporal_list = []
        loss_cl_list = []
        for shot_data in batch_data: #clip data
            # read input data
            img_features, text_features, gt = shot_data

            # processed_img = processor(images=imgs, return_tensors='pt').pixel_values.to(device)

            # process model
            optim.zero_grad()
            
            output, loss_shot, loss_temp, loss_cl = net([[i.to(device) for i in img_features], [i.to(device) for i in text_features]], {
                "temp": gt[1],
                "shot": None,
                "scene": None
            })
            # loss_shot = loss_func(output, gt)
            # PRED = pred_func(output)
            PRED = output
            # GT = gt
            score_shot = int(PRED == gt[1])
            # print(PRED, GT)
            loss_batch_list.append(loss_shot)
            score_batch_list.append(score_shot)
            loss_temporal_list.append(loss_temp)
            loss_cl_list.append(loss_cl)


        # calcuclate avearge batch
        score_step = sum(score_batch_list) / len(score_batch_list)
        loss_step = sum(loss_batch_list) / len(loss_batch_list)
        loss_step.backward()
        optim.step()
        # caculate avearge score
        score_epoch_list.append(score_step)
        loss_epoch_list.append(float(loss_step))
        wandb.log({
            'train loss':sum(loss_epoch_list)/len(loss_epoch_list), 
            'train score':sum(score_epoch_list)/len(score_epoch_list),
            'temp loss': sum(loss_temporal_list) / len(loss_temporal_list),
            'cl loss': sum(loss_cl_list) / len(loss_cl_list)
        })

    score_epoch = sum(score_epoch_list) / len(score_epoch_list)
    loss_epoch = sum(loss_epoch_list) / len(loss_epoch_list)
    print('train loss = ', loss_epoch, 'train score = ', score_epoch)  

    # val
    net.eval()
    with torch.no_grad():
        loss_epoch_list = []
        score_epoch_list = []
        cl_epoch_list = []
        for batch_data in tqdm(val_dataloader):
            loss_batch_list = []
            score_batch_list = []

            cl_score_list = []

            for shot_data in batch_data: #clip data
                # read input data
                img_features, text_features, gt = shot_data

                output, loss_shot, cl_score = net([[i.to(device) for i in img_features], [i.to(device) for i in text_features]])
                # loss_shot = loss_func(output, gt)W
                # PRED = pred_func(output)
                PRED = output
                # GT = gt
                score_shot = int(PRED == gt[1])
                # print(PRED, GT)
                loss_batch_list.append(loss_shot)
                score_batch_list.append(score_shot)
                cl_score_list.append(cl_score)
                
            # calcuclate avearge batch
            score_step = sum(score_batch_list) / len(score_batch_list)
            loss_step = sum(loss_batch_list) / len(loss_batch_list)
            cl_step = sum(cl_score_list) / len(cl_score_list)

            # caculate avearge score
            score_epoch_list.append(score_step)
            cl_epoch_list.append(cl_step)
            loss_epoch_list.append(float(loss_step))
            wandb.log({'val loss':sum(loss_epoch_list)/len(loss_epoch_list), 'val score':sum(score_epoch_list)/len(score_epoch_list)})

        score_epoch = sum(score_epoch_list) / len(score_epoch_list)
        loss_epoch = sum(loss_epoch_list) / len(loss_epoch_list)
        cl_epoch = sum(cl_epoch_list) / len(cl_epoch_list)
        print('val loss = ', loss_epoch, 'val score = ', score_epoch, 'val cl score = ', cl_epoch)
        if score_epoch >= best_val_acc: 
            best_val_acc = score_epoch
            torch.save(net.state_dict(), Path('./checkpoint', f'frame_to_shot_best_{timestamp}.pth'))
            print("save epoch ",e)
