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
import requests
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

data_path = '/home/jianghui/dataset/VideoReorder-MovieNet'

train_data = VideoReorderMovieNetDataFolder(root=data_path, split='train')
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, collate_fn=lambda x: x)


val_data = VideoReorderMovieNetDataFolder(root=data_path, split='val')
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=lambda x: x)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
net.to(device)

lr = 1e-4
epoch = 40

loss_func = PairWiseLoss()
loss_func.to(device)
pred_func = PairWisePred()
pred_func.to(device)
optim = torch.optim.AdamW(net.parameters(), lr=lr)

for e in range(epoch):
    print(f'epoch {e}:')
    running_loss = 0.0

    # train
    for i, data in tqdm(enumerate(train_dataloader)):
        # print(i, end=" ")
        for img_list, text_list in data:
            feature_list = []
            for img, text in zip(img_list, text_list):

                processed_img = processor(images=img, return_tensors='pt').to(device)
                image_features = model.get_image_features(**processed_img)
                tokenizered_txt = tokenizer(text=text, padding=True, return_tensors="pt").to(device)
                text_features = model.get_text_features(**tokenizered_txt)

                all_feature = torch.cat((image_features, text_features), dim=1).to(device)
                feature_list.append(all_feature)
            
            optim.zero_grad()

            output_list = []
            for input in feature_list:
                outputs = net(input)
                output_list.append(outputs)

            output_tensor= torch.stack(output_list, dim = 0)
            
            loss = loss_func(output_tensor)
            loss.backward()
            optim.step()

            
            running_loss += loss
    print(f'running_loss in one epoch is {running_loss / i}')

    # val
    for i, data in tqdm(enumerate(val_dataloader)):
        # print(i, end=" ")
        score_list = []
        for img_list, text_list in data:
            feature_list = []
            for img, text in zip(img_list, text_list):

                processed_img = processor(images=img, return_tensors='pt').to(device)
                image_features = model.get_image_features(**processed_img)
                tokenizered_txt = tokenizer(text=text, padding=True, return_tensors="pt").to(device)
                text_features = model.get_text_features(**tokenizered_txt)

                all_feature = torch.cat((image_features, text_features), dim=1).to(device)
                feature_list.append(all_feature)

            output_list = []
            for input in feature_list:
                outputs = net(input)
                output_list.append(outputs)

            # output_tensor= torch.stack(output_list, dim = 0)
            pred_list = pred_func(output_list)
            score = TripleLengthMatching(pred_list, GT=[])
            # print(f'once score is {score}')
            score_list.append(score)
        print(f'average score is {sum(score_list)/len(score_list)}')