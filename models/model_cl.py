import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertConfig
import copy

class OneLayer_CL(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.num_shot_cls = 4 # 2 frame in a shot and 3 negative frame from other shot
        self.lambda_shot_cls = 0.0
        self.lambda_scene_cls = 0.0

        # 1 init
        # transformer_block = BertModel.from_pretrained('path', num_hidden_layers=1)
        encoder_config = BertConfig(num_hidden_layers=1, hidden_size=512, num_attention_heads=8)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        # 2.1 embedding layer
        self.img_embedding = nn.Sequential(
            # nn.Linear(768, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 768)
            nn.Linear(768, 512)
        )
        self.text_embedding = nn.Sequential(
            # nn.Linear(512, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 768)
            nn.Linear(512, 512),
            nn.Dropout(0.1, inplace=True)            
        )

        self.img_embedding.apply(init_weights)
        self.text_embedding.apply(init_weights)

        # 2.2 encoder
        self.encoder_transformer_blocked = BertModel(config=encoder_config)

        # 2.3 cls_mlp
        self.value_fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
        self.value_fn.apply(init_weights)


        self.shot_cls_fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.num_shot_cls)
        )

        self.loss_temporal = torch.nn.LogSigmoid()
        self.loss_shot_cls = torch.nn.CrossEntropyLoss()

    
    def forward(self, input, gt=None):
        img_tokens, text_tokens = input

        img0, img1, img2, img3, img4 = img_tokens
        text0, text1, text2, text3, text4 = text_tokens
        
        # embedding layer
        B = img0.size(0)

        img0_embed = self.img_embedding(img0)
        img1_embed = self.img_embedding(img1)
        img2_embed = self.img_embedding(img2)
        img3_embed = self.img_embedding(img3)
        img4_embed = self.img_embedding(img4)


        text0_embed = self.text_embedding(text0)
        text1_embed = self.text_embedding(text1)
        text2_embed = self.text_embedding(text2)
        text3_embed = self.text_embedding(text3)
        text4_embed = self.text_embedding(text4)


        # encoder
        encoder_input0 = torch.cat((text0_embed, img0_embed), dim=1)
        encoder_input1 = torch.cat((text1_embed, img1_embed), dim=1)
        encoder_input2 = torch.cat((text2_embed, img2_embed), dim=1)
        encoder_input3 = torch.cat((text3_embed, img3_embed), dim=1)
        encoder_input4 = torch.cat((text4_embed, img4_embed), dim=1)

        pos0, pos1 = 0, text0_embed.size(1)
        encoder_output0 = self.encoder_transformer_blocked(inputs_embeds=encoder_input0)[0] # B, text_len + path_len=132 , 768
        encoder_output1 = self.encoder_transformer_blocked(inputs_embeds=encoder_input1)[0]
        encoder_output2 = self.encoder_transformer_blocked(inputs_embeds=encoder_input2)[0]
        encoder_output3 = self.encoder_transformer_blocked(inputs_embeds=encoder_input3)[0]
        encoder_output4 = self.encoder_transformer_blocked(inputs_embeds=encoder_input4)[0]

        # return torch.stack([encoder_output[:,pos0], encoder_output[:, pos1]], dim=0)

        value0 = self.value_fn(torch.cat([encoder_output0[:, 0], encoder_output0[:, text0_embed.size(1)]], dim=1).view(B, 2 * 512))
        value1 = self.value_fn(torch.cat([encoder_output1[:, 0], encoder_output1[:, text1_embed.size(1)]], dim=1).view(B, 2 * 512))
        value1 = self.value_fn(torch.cat([encoder_output2[:, 0], encoder_output1[:, text2_embed.size(1)]], dim=1).view(B, 2 * 512))
        value1 = self.value_fn(torch.cat([encoder_output3[:, 0], encoder_output1[:, text3_embed.size(1)]], dim=1).view(B, 2 * 512))
        value1 = self.value_fn(torch.cat([encoder_output4[:, 0], encoder_output1[:, text4_embed.size(1)]], dim=1).view(B, 2 * 512))

        # cls_mlp
        # cls_input = torch.cat([img0[:, 0], img1[:, 0]], dim=-1)
        # cls_input = torch.cat([text0[:, 0], text1[:, 0]], dim=-1)
        # cls_input = torch.cat((encoder_output[:, pos0], encoder_output[:, pos1]), dim=-1) # B, 768*2
        # cls_input = encoder_output[:,0,:]
        shot_cls = 0.0

        loss_temporal = 0.0
        if gt is not None:
            if gt["temp"] == 1:
                loss_temporal = - self.loss_temporal(value1 - value0)
            else:
                loss_temporal = - self.loss_temporal(value0 - value1)

        loss_shot = loss_temporal 


        return (value1 > value0).int().cpu(), loss_shot
    

class OneLayer_CL_infer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.num_shot_cls = 1000
        self.num_scene_cls = 1000
        self.lambda_shot_cls = 0.0
        self.lambda_scene_cls = 0.0

        # 1 init
        # transformer_block = BertModel.from_pretrained('path', num_hidden_layers=1)
        encoder_config = BertConfig(num_hidden_layers=1, hidden_size=512, num_attention_heads=8)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        # 2.1 embedding layer
        # self.class0_embedding = nn.Parameter(torch.randn(768))
        # self.class1_embedding = nn.Parameter(torch.randn(768))
        self.img_embedding = nn.Sequential(
            # nn.Linear(768, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 768)
            nn.Linear(768, 512)
        )
        self.text_embedding = nn.Sequential(
            # nn.Linear(512, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 768)
            nn.Linear(512, 512),
            nn.Dropout(0.1, inplace=True)            
        )

        self.img_embedding.apply(init_weights)
        self.text_embedding.apply(init_weights)

        # 2.2 encoder
        self.encoder_transformer_blocked = BertModel(config=encoder_config)

        # 2.3 cls_mlp
        self.value_fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
        self.value_fn.apply(init_weights)


        self.shot_cls_fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.num_shot_cls)
        )

        self.loss_temporal = torch.nn.LogSigmoid()
        self.loss_shot_cls = torch.nn.CrossEntropyLoss()

    
    def forward(self, input, gt=None):
        img_tokens, text_tokens = input
        img0, img1 = img_tokens
        text0, text1 = text_tokens
        
        # embedding layer
        B = img0.size(0)
        img0_embed = self.img_embedding(img0)
        # class0_embeds = self.class0_embedding.expand(B, 1, -1)
        # img0_embed = torch.cat([class0_embeds, img0_embed], dim=1)
        img1_embed = self.img_embedding(img1)
        # class1_embeds = self.class1_embedding.expand(B, 1, -1)
        # img1_embed = torch.cat([class1_embeds, img1_embed], dim=1)
        text0_embed = self.text_embedding(text0)
        text1_embed = self.text_embedding(text1)

        # encoder
        encoder_input1 = torch.cat((text0_embed, img0_embed), dim=1)
        encoder_input2 = torch.cat((text1_embed, img1_embed), dim=1)

        pos0, pos1 = 0, text0_embed.size(1)
        encoder_output1 = self.encoder_transformer_blocked(inputs_embeds=encoder_input1)[0] # B, text_len + path_len=132 , 768
        encoder_output2 = self.encoder_transformer_blocked(inputs_embeds=encoder_input2)[0]
        # return torch.stack([encoder_output[:,pos0], encoder_output[:, pos1]], dim=0)


        value1 = self.value_fn(torch.cat([encoder_output1[:, 0], encoder_output1[:, text0_embed.size(1)]], dim=1).view(B, 2 * 512))
        value2 = self.value_fn(torch.cat([encoder_output2[:, 0], encoder_output2[:, text1_embed.size(1)]], dim=1).view(B, 2 * 512))


        # cls_mlp
        # cls_input = torch.cat([img0[:, 0], img1[:, 0]], dim=-1)
        # cls_input = torch.cat([text0[:, 0], text1[:, 0]], dim=-1)
        # cls_input = torch.cat((encoder_output[:, pos0], encoder_output[:, pos1]), dim=-1) # B, 768*2
        # cls_input = encoder_output[:,0,:]

        return value1, value2
        shot_cls = 0.0

        loss_temporal = 0.0
        if gt is not None:
            if gt["temp"] == 1:
                loss_temporal = - self.loss_temporal(value2 - value1)
            else:
                loss_temporal = - self.loss_temporal(value1 - value2)

        loss_shot = loss_temporal 


        return (value2 > value1).int().cpu(), loss_shot