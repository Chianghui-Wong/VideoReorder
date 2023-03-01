import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertConfig, CLIPModel
import copy


class OneLayer_new(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # bert_model = BertModel.from_pretrained(num_hidden_layers=1)
        

        # # 1 init
        # # transformer_block = BertModel.from_pretrained('path', num_hidden_layers=1)
        # encoder_config = BertConfig(num_hidden_layers=1, hidden_size=768)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        # # 2.1 embedding layer
        # self.class0_embedding = nn.Parameter(torch.randn(768))
        # self.class1_embedding = nn.Parameter(torch.randn(768))
        # self.img_embedding = nn.Sequential(
        #     # nn.Linear(768, 1024),
        #     # nn.ReLU(),
        #     # nn.Linear(1024, 768)
        #     nn.Linear(768, 768)
        # )
        # self.text_embedding = nn.Sequential(
        #     # nn.Linear(512, 1024),
        #     # nn.ReLU(),
        #     # nn.Linear(1024, 768)
        #     nn.Linear(512, 768)            
        # )

        # self.img_embedding.apply(init_weights)
        # self.text_embedding.apply(init_weights)

        # 2.2 encoder
        encoder_config = BertConfig(num_hidden_layers=1, hidden_size=768)
        self.encoder_transformer_blocked = BertModel(config=encoder_config)

        # 2.3 cls_mlp
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768*2, 768),
            nn.Tanh(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768,2)
        )
        self.cls.apply(init_weights)

    
    def forward(self, input):

        
        img_tokens, text_tokens = input
        img0, img1 = img_tokens
        text0, text1 = text_tokens
        
        


        # embedding layer
        B = img0.size(0)
        img0_embed = self.img_embedding(img0)
        class0_embeds = self.class0_embedding.expand(B, 1, -1)
        img0_embed = torch.cat([class0_embeds, img0_embed], dim=1)
        img1_embed = self.img_embedding(img1)
        class1_embeds = self.class1_embedding.expand(B, 1, -1)
        img1_embed = torch.cat([class1_embeds, img1_embed], dim=1)
        text0_embed = self.text_embedding(text0)
        text1_embed = self.text_embedding(text1)

        # encoder
        encoder_input = torch.cat((img0_embed, text0_embed, img1_embed, text1_embed), dim=1)
        pos0, pos1 = 0, img0_embed.size(1)+text0_embed.size(1)
        encoder_output = self.encoder_transformer_blocked(inputs_embeds=encoder_input)[0] # B, text_len + path_len=132 , 768

        # return torch.stack([encoder_output[:,pos0], encoder_output[:, pos1]], dim=0)

        # cls_mlp
        # cls_input = torch.cat([img0[:, 0], img1[:, 0]], dim=-1)
        cls_input = torch.cat((encoder_output[:, pos0], encoder_output[:, pos1]), dim=-1) # B, 768*2
        # cls_input = encoder_output[:,0,:]
        cls_output = self.cls(cls_input)

        return cls_output