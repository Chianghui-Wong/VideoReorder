import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertConfig

class OneLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # 1 init
        # transformer_block = BertModel.from_pretrained('path', num_hidden_layers=1)
        img_config = BertConfig(num_hidden_layers=1, hidden_size=768)
        text_config = BertConfig(num_hidden_layers=1, hidden_size=512)
        encoder_config = BertConfig(num_hidden_layers=1, hidden_size=1024)

        # 2.1 embedding layer
        self.img_transformer_blocked = BertModel(config=img_config)
        self.text_transformer_blocked = BertModel(config=text_config)

        # 2.2 encoder
        self.encoder_transformer_blocked = BertModel(config=encoder_config)

        # 2.3 cls_mlp
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,2)
        )

    
    def forward(self, input):
        img0, text0, img1, text1 = input
        
        # embedding layer
        img0_embed = self.img_transformer_blocked(img0)[0]
        img1_embed = self.img_transformer_blocked(img1)[0]
        text0_embed = self.text_transformer_blocked(text0)[0]
        text1_embed = self.text_transformer_blocked(text1)[0]

        # encoder
        input0 = torch.cat((img0_embed, text0_embed), dim=1)
        input1 = torch.cat((img1_embed, text1_embed), dim=1)
        output0 = self.encoder_transformer_blocked(encoder_hidden_states=input0)[0], # B, text_len + path_len , 768
        output1 = self.encoder_transformer_blocked(encoder_hidden_states=input1)[1]

        # cls_mlp
        cls_input = torch.cat((output0, output1), dim=1)
        cls_output = self.cls(cls_input)

        return cls_output