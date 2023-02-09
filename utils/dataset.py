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
            root: str,
            transform: Optional[Callable] = None
        ) -> None:

        return None
    
    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        image_list = 0
        text_list = 1
        return image_list, text_list

class VideoReorderMovieNetDataLoader(object):
    def __init__(self) -> None:
        pass

    def __next__(self):
        return


def build_VideoReorderMovieNet_dataset(args):
    transform = DataAugmentationForVRM(args)
    print("Data Aug = %s" % str(transform))
    return VideoReorderMovieNetDataFolder(args.data_path, transform=transform)