# This file is use to read Hierachal-MovieNet dataset
import numpy as np
import torch
import cv2
import json
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import os
import random
from tqdm import tqdm
import sys
from .tools import *


class DataAugmentationForVRM(object):
    # VRM : Video Reorder MovieNet
    def __init__(self, args):
        pass

    def __call__(self, image, text):
        return image, text

    def __repr__(self):
        repr = "(DataAugmentationForVRM)"
        return repr

class VideoReorderMovieNetDataFolder(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str, # = '/home/jianghui/dataset/VideoReorder-MovieNet'
            split: str,
            layer: str = '',
            transform: Optional[Callable] = None
        ) -> None:
        
        super().__init__()
        # init
        if split == 'test': self.split = 'test_in_domain'
        if split not in ['train', 'val', 'test_in_domain', 'test_out_domain', 'all', 'human_behavior/in_domain', "human_behavior/out_domain"]: assert False, 'No such split name'
        self.split = split
        
        self.root = Path(root)

        if layer not in ['', 'frame', 'shot', 'scene', 'all', 'ori'] : assert False, 'No such layer name'
        self.layer = layer

        # read data .pt file
        if self.layer == 'ori':
            with open(Path(self.root, 'clip_id.json'), 'r') as fr:
                clip_id_json = json.load(fr)
            self.clip_id_list = clip_id_json[self.split]

        elif self.layer == '':
            self.data = torch.load(Path(self.root, f'{split}.pt'))
        else:
            self.data = torch.load(Path(self.root, f'{split}_{self.layer}.pt'))

        return
    
    def __len__(self):
        if self.layer == 'ori':
            return len(self.clip_id_list)
        else:
            return len(self.data)
    
    def __getitem__(self, index):
        if self.layer == "":
            return self.data[index]['img_features'], self.data[index]['text_features'], self.data[index]['gt_id'], self.data[index]['shot_id'], self.data[index]['scene_id']
        
        if self.layer in ['frame', 'shot', 'scene']:
            return self.data[index]['img_features'], self.data[index]['text_features'], self.data[index]['gt_id']
        
        if self.layer in ['ori']:
            clip_id = self.clip_id_list[index]
            img_list = []
            text_list = []
            clip_path = Path(self.root, self.split, str(clip_id))

            with open(Path(clip_path, 'info.json'), 'r') as f_r:
                info_shuffled_json = json.load(f_r)
            
            # read text
            with open(Path(clip_path, 'subtitle.json'), 'r') as f_r:
                subtitle_json = json.load(f_r)

            # read img
            for img_id in info_shuffled_json['img_id']:
                img_ele = Image.open(Path(clip_path, f'{clip_id}_{img_id}.jpg'))
                img_list.append(img_ele)
                text_list.append(subtitle_json[img_id])

            return img_list, text_list




            
class VideoReorderMovieNetDataLoader(object):
    def __init__(self) -> None:
        pass

    def __next__(self):
        return


def build_VideoReorderMovieNet_dataset(args):
    '''
    return  list[list[], ...]
    the fisrt list is batch
    the second is [['feature'], ['shot_id'], ...]
    '''
    transform = DataAugmentationForVRM(args)
    print("Data Aug = %s" % str(transform))
    root = args.data_path
    split = args.split
    return VideoReorderMovieNetDataFolder(root, split, transform=transform, collate_fn=lambda x: x)

if __name__ == '__main__':
    pass