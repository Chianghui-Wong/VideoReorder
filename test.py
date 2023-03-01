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
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn
import time
from scipy.special import comb, perm

# model = OneLayer()

img0 = torch.rand(1, 50, 768)
img1 = torch.rand(1, 50, 768)

text0 = torch.rand(1, 17, 512)
text1 = torch.rand(1, 16, 512)

output = torch.cat([text0, text1], dim=1)
print(output.shape)


# tmp = text0[:,0,:]
# print(tmp.shape)
# output = model([[img0, text0], [img1, text1]])
# print(output)

# LossFunc = ClipPairWiseLoss()
# input = torch.tensor([[4, 4], [3, 2]]).float()
# loss = LossFunc(input, [1, 0])
# print(loss)

# data_path = '/home/jianghui/dataset/VideoReorder-MovieNet'
# split = 'train'
# train_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='ori')
# train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=(split == 'train'), num_workers=8, pin_memory=True, collate_fn=lambda x: x)

# print(get_order_index([1.2, 0.8]))

# print(get_order_list([1.2, 0.8]))