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
        
        # init
        super().__init__()
        if split == 'test': self.split = 'test_in_domain'
        
        if split in ['train', 'val', 'test_in_domain', 'test_out_domain', 'all', 'human_behavior/in_domain', "human_behavior/out_domain"]:
            self.split = split
        else:
            assert False, 'No such split name '
        
        self.root = Path(root)
        self.layer = layer

        # read clip_id.json
        with open(Path(self.root, 'clip_id.json'), 'r') as f:
            clip_id_json = json.load(f)
        
        self.clip_list = clip_id_json[self.split]

        # read data .pt file
        self.data = torch.load(Path(self.root, f'{split}{self.layer}.pt'))

        return
    
    def __len__(self):
        return len(self.clip_list)
    
    def __getitem__(self, index):
        clip_id = self.clip_list[index]

        return self.data[clip_id]['feature'], self.data[clip_id]['img_id'], self.data[clip_id]['shot_id'], self.data[clip_id]['scene_id']

class VideoReorderMovieNetDataLoader(object):
    def __init__(self) -> None:
        pass

    def __next__(self):
        return


def build_VideoReorderMovieNet_dataset(args):
    '''
    return  list[list[], ...]
    the fisrt list is batch
    the second is [['feature'], ['shot_id'], ['scene_id]]
    '''
    transform = DataAugmentationForVRM(args)
    print("Data Aug = %s" % str(transform))
    root = args.data_path
    split = args.split
    return VideoReorderMovieNetDataFolder(root, split, transform=transform, collate_fn=lambda x: x)

if __name__ == '__main__':
    pass