# common lib
import argparse, sys, os, io, base64, pickle, json, math
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch 
import torchvision 
import cv2
from PIL import Image
import transformers
import copy
import yaml
from pathlib import Path
import types
# backbone lib
from dataset import Dataset_Base
from model import VIOLET_Base
from agent import Agent_Base
# my lib
from utils import *

# environment
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser('Equi-Separation training script', add_help=False)

    # Data
    parser.add_argument('--dataset_dir', default='./data/', type=str)
    parser.add_argument('--class_num', default=10, type=int)
    parser.add_argument('--color_channel', default=1, type=int)

    # Architecture
    parser.add_argument('--model', default='GFNN', type=str)
    parser.add_argument('--layer_num', default=2, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--input_size', default=100, type=int)

    # Train
    parser.add_argument('--measure', default='within_variance', type=str)
    parser.add_argument('--optimization', default='adam', type=str)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--simple_train_batch_size', default=128, type=int)
    parser.add_argument('--simple_test_batch_size', default=100, type=int)
    parser.add_argument('--epoch_num', default=600, type=int)
    parser.add_argument('--momentum', default=0.9, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--ample_size_per_class', default=100, type=int)
    parser.add_argument('--normalization', default=None, type=str)
    parser.add_argument('--eps', default=None, type=str)

    # Other
    parser.add_argument('--outputs_dir', default='./outputs/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=str)
    parser.add_argument('--figure_dir', default='./figure/', type=str)
    parser.add_argument('--config', default='./configs/frame_reorder.yaml', type=str)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    return args

def main(args):
    print(args)
    pass

if __name__ == '__main__':
    opts = get_args()
    main(opts)