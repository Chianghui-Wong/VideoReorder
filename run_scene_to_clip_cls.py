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
    name = 'scene to clip cls'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

data_path = '/home/jianghui/dataset/VideoReorder-MovieNet'
split = 'train'
train_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='scene')
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=(split == 'train'), num_workers=0, pin_memory=True, collate_fn=lambda x: x)

split = 'val'
val_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='scene')
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=(split == 'train'), num_workers=0, pin_memory=True, collate_fn=lambda x: x)

net = OneLayer()
net.to(device)
wandb.watch(net)

lr = 1e-5
epoch = 5

# loss_func = ClipPairWiseLoss()
# loss_func.to(device)
loss_func = torch.nn.CrossEntropyLoss()
pred_func = ClipPairWisePred()
pred_func.to(device)
optim = torch.optim.AdamW(net.parameters(), lr=lr)


best_val_acc = 0.0
for e in range(epoch):
    print(f'epoch {e}:')

    # train
    net.train()
    epoch_record = {
        'score':0.0, 
        'loss':0.0,
        'num':0.0,
        }
    for batch_data in tqdm(train_dataloader):
        batch_record = {
            'score':0.0, 
            'loss':0.0,
            'num':0.0,
            }

        for pair_data in batch_data: #clip data
            # read input data
            img_features, text_features, gt = pair_data

            # processed_img = processor(images=imgs, return_tensors='pt').pixel_values.to(device)

            # process model
            optim.zero_grad()
            
            output = net([[img_features[0].to(device), img_features[1].to(device)], [text_features[0].to(device), text_features[1].to(device)]])
            # loss_shot = loss_func(output, gt)
            loss_shot = loss_func(output, torch.tensor([gt[1]]).to(device))
            # PRED = pred_func(output)
            PRED = get_order_list(output.reshape(-1).cpu())
            GT = gt
            score_shot = int(PRED == gt)
            # print(PRED, GT)
            batch_record['loss'] += loss_shot
            batch_record['score'] += score_shot
            batch_record['num'] += 1
        
        # calcuclate avearge batch
        batch_record['loss'] /= batch_record['num']
        batch_record['score'] /= batch_record['num']
        score_step = batch_record['score']
        loss_step = batch_record['loss']
        wandb.log({'train loss':float(loss_step), 'train score':score_step})

        # back & optim
        loss_step.backward()
        optim.step()

        loss_step = float(loss_step)
        batch_record['loss'] = float(batch_record['loss'])

        # caculate avearge score
        epoch_record['loss'] += float(loss_step)
        epoch_record['score'] += score_step
        epoch_record['num'] += 1

    score_epoch = epoch_record['score'] / epoch_record['num']
    loss_epoch = epoch_record['loss'] / epoch_record['num']
    print('train loss = ', loss_epoch, 'train score = ', score_epoch)  

    # val
    net.eval()
    with torch.no_grad():
        epoch_record = {
            'score':0.0, 
            'loss':0.0,
            'num':0.0,
            }
        for batch_data in tqdm(val_dataloader):
            batch_record = {
                'score':0.0, 
                'loss':0.0,
                'num':0.0,
                }

            for pair_data in batch_data: #clip data
                # read input data
                img_features, text_features, gt = pair_data

                output = net([[img_features[0].to(device), img_features[1].to(device)], [text_features[0].to(device), text_features[1].to(device)]])
                # loss_shot = loss_func(output, gt)
                loss_shot = loss_func(output, torch.tensor([gt[1]]).to(device))

                # PRED = pred_func(output)
                PRED = get_order_list(output.reshape(-1).cpu())
                GT = gt
                score_shot = int(PRED == gt)
                # print(PRED, GT)
                batch_record['loss'] += loss_shot
                batch_record['score'] += score_shot
                batch_record['num'] += 1
                
            # calcuclate avearge batch
            batch_record['loss'] /= batch_record['num']
            batch_record['score'] /= batch_record['num']
            score_step = batch_record['score']
            loss_step = batch_record['loss']
            wandb.log({'val loss':float(loss_step), 'val score':score_step})

            # caculate avearge score
            epoch_record['loss'] += float(loss_step)
            epoch_record['score'] += score_step
            epoch_record['num'] += 1

        score_epoch = epoch_record['score'] / epoch_record['num']
        loss_epoch = epoch_record['loss'] / epoch_record['num']
        print('val loss = ', loss_epoch, 'val score = ', score_epoch)

        if score_epoch >= best_val_acc: 
            best_val_acc = score_epoch
            torch.save(net.state_dict(), Path('./checkpoint', f'scene_to_clip_best_{timestamp}.pth'))
            print("save epoch ",e)
