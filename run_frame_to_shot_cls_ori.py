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
import random, itertools

from utils import *
from models import *

from transformers import AutoProcessor, AutoTokenizer, CLIPModel, BertTokenizer, AlbertModel
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


timestamp = time.strftime('%Y-%m-%d', time.localtime(time.time()))
wandb.init(
    project = 'VideoReorder',
    name = 'frame to shot cls'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

data_path = '/home/jianghui/dataset/VideoReorder-MovieNet'
split = 'train'
train_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='frame')
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=(split == 'train'), num_workers=8, pin_memory=True, collate_fn=lambda x: x)

split = 'val'
val_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='frame')
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=(split == 'train'), num_workers=8, pin_memory=True, collate_fn=lambda x: x)

net = OneLayer_new()
net.to(device)
wandb.watch(net)

lr = 1e-5
epoch = 40

# loss_func = ClipPairWiseLoss()
# loss_func.to(device)
loss_func = torch.nn.CrossEntropyLoss()
pred_func = ClipPairWisePred()
pred_func.to(device)
optim = torch.optim.AdamW(net.parameters(), lr=lr)


# init embedding layer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
text_embed = clip_model.text_model.embeddings
vision_embed = clip_model.vision_model.embeddings


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

        for shot_data in batch_data: #clip data
            # read input data
            imgs, texts = shot_data

            processed_img = processor(images=imgs, return_tensors='pt').pixel_values.to(device)
            processed_text = tokenizer(text=texts, return_tensors='pt').to(device)

            

            L = processed_img.size(0) 
            coups = itertools.combinations(range(L), 2)

            for (idx1, idx2) in coups:
                

                img_pair = torch.cat((processed_img[idx1], processed_img[idx2]), dim=0)
                img_feature = vision_embed(img_pair)
                text_pair = torch.cat((processed_text[idx1], processed_text[idx2]), dim=0)


                output = net([[processed_img[idx1], processed_img[idx2]], [processed_text[idx1], processed_text[idx2]]])
        

            # process model
            optim.zero_grad()
            
            output = net([])

            output = net([[img_features[0].to(device), img_features[1].to(device)], [text_features[0].to(device), text_features[1].to(device)]])
            # loss_shot = loss_func(output, gt)
            loss_shot = loss_func(output, torch.tensor([gt[0]]).to(device))
            # PRED = pred_func(output)
            PRED = get_order_list(output.reshape(-1).cpu())
            GT = gt
            score_shot = int(PRED == gt)
            # print(PRED, GT)
            loss_batch_list.append(loss_shot)
            score_batch_list.append(score_shot)
            
        # calcuclate avearge batch
        score_step = sum(score_batch_list) / len(score_batch_list)
        loss_step = sum(loss_batch_list) / len(loss_batch_list)
        loss_step.backward()
        optim.step()
        # caculate avearge score
        score_epoch_list.append(score_step)
        loss_epoch_list.append(float(loss_step))
        wandb.log({'train loss':sum(loss_epoch_list)/len(loss_epoch_list), 'train score':sum(score_epoch_list)/len(score_epoch_list)})

    score_epoch = sum(score_epoch_list) / len(score_epoch_list)
    loss_epoch = sum(loss_epoch_list) / len(loss_epoch_list)
    print('train loss = ', loss_epoch, 'train score = ', score_epoch)  

    # val
    net.eval()
    with torch.no_grad():
        loss_epoch_list = []
        score_epoch_list = []
        for batch_data in tqdm(val_dataloader):
            loss_batch_list = []
            score_batch_list = []

            for shot_data in batch_data: #clip data
                # read input data
                img_features, text_features, gt = shot_data

                output = net([[img_features[0].to(device), img_features[1].to(device)], [text_features[0].to(device), text_features[1].to(device)]])
                # loss_shot = loss_func(output, gt)
                loss_shot = loss_func(output, torch.tensor([gt[0]]).to(device))

                # PRED = pred_func(output)
                PRED = get_order_list(output.reshape(-1).cpu())
                GT = gt
                score_shot = int(PRED == gt)
                # print(PRED, GT)
                loss_batch_list.append(loss_shot)
                score_batch_list.append(score_shot)
                
            # calcuclate avearge batch
            score_step = sum(score_batch_list) / len(score_batch_list)
            loss_step = sum(loss_batch_list) / len(loss_batch_list)

            # caculate avearge score
            score_epoch_list.append(score_step)
            loss_epoch_list.append(float(loss_step))
            wandb.log({'val loss':sum(loss_epoch_list)/len(loss_epoch_list), 'val score':sum(score_epoch_list)/len(score_epoch_list)})

        score_epoch = sum(score_epoch_list) / len(score_epoch_list)
        loss_epoch = sum(loss_epoch_list) / len(loss_epoch_list)
        print('val loss = ', loss_epoch, 'val score = ', score_epoch)
        if score_epoch >= best_val_acc: 
            best_val_acc = score_epoch
            torch.save(net.state_dict(), Path('./checkpoint', f'frame_to_shot_best_{timestamp}.pth'))
            print("save epoch ",e)
