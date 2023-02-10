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
import tqdm


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
            transform: Optional[Callable] = None
        ) -> None:
        
        # init
        super().__init__()
        if split == 'test': self.split = 'test_in_domain'
        
        if split in ['train', 'val', 'test_in_domain', 'test_out_domain', 'all', 'human_behavior/in_domain', "human_behavior/out_domain"]:
            self.split = split
        else:
            assert False, 'No such split name in MovieNet'
        
        self.root = Path(root)

        # read clip file name
        with open(Path(self.root, 'clip_id.json'), 'r') as f:
            clip_id_json = json.load(f)
        
        self.clip_list = clip_id_json[self.split]

        return None
    
    def __len__(self):
        return len(self.clip_list)
    
    def __getitem__(self, index):
        clip_id = self.clip_list[index]
        clip_path = Path(self.root, self.split, str(clip_id))

        with open(Path(clip_path, "info.json"), 'r') as f:
            info_json = json.load(f)
        
        img_list = []
        for idx in info_json['img_id']:
            img_tmp = Image.open(Path(clip_path, f'{clip_id}_{idx}.jpg'))
            img_list.append(img_tmp)
        
        with open(Path(clip_path, 'subtitle.json'), 'r') as f:
            text_list = json.load(f)

        return img_list, text_list

class VideoReorderMovieNetDataLoader(object):
    def __init__(self) -> None:
        pass

    def __next__(self):
        return


def build_VideoReorderMovieNet_dataset(args):
    transform = DataAugmentationForVRM(args)
    print("Data Aug = %s" % str(transform))
    root = args.data_path
    split = args.split
    return VideoReorderMovieNetDataFolder(root, split, transform=transform)

if __name__ == '__main__':
    data_path = '/home/jianghui/dataset/VideoReorder-MovieNet'
    split = 'train'
    train_data = VideoReorderMovieNetDataFolder(data_path, split)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=(split == 'train'), num_workers=1, pin_memory=True, collate_fn=lambda x: x)

    for data in tqdm(train_dataloader):
        # print("data shape is", np.array(data).shape)
        for img_list, text_list in data:
            for img, text in zip(img_list, text_list):
                print(img)
                print(text)

        assert False    