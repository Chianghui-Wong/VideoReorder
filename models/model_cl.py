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

        self.shot_hidden_size = 512 # 2 frame in a shot and 3 negative frame from other shot
        self.lambda_cl = 0.01
        self.lambda_scene_cls = 0.0

        # 1 init
        # transformer_block = BertModel.from_pretrained('path', num_hidden_layers=1)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        # 2.1 embedding layer
        self.img_embedding = nn.Sequential(
            # nn.Linear(768, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 768)
            nn.Linear(768, 512, bias=False)
        )
        self.text_embedding = nn.Sequential(
            # nn.Linear(512, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 768)
            nn.Linear(512, 512, bias=False),
            # nn.Dropout(0.1, inplace=True)            
        )

        self.img_embedding.apply(init_weights)
        self.text_embedding.apply(init_weights)

        # 2.2 encoder
        encoder_config = BertConfig(num_hidden_layers=2, hidden_size=512, num_attention_heads=8)
        self.encoder_transformer_blocked = BertModel(config=encoder_config)
        self.encoder_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2, 512, bias=False)
        )

        # 2.3 cls_mlp
        self.value_fn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512,1, bias=False)
        )
        self.value_fn.apply(init_weights)


        self.shot_cls_fn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, self.shot_hidden_size, bias=False)
        )
        self.shot_cls_fn.apply(init_weights)

        self.loss_temporal = torch.nn.LogSigmoid()
        self.loss_shot_cls = torch.nn.CrossEntropyLoss()


    def encode(self, img, text):
        B = img.size(0)
        img_embed = self.img_embedding(img)
        text_embed = self.text_embedding(text)

        encoder_input = torch.cat((img_embed, text_embed), dim=1)
        encoder_output = self.encoder_transformer_blocked(inputs_embeds=encoder_input)[0]

        encoder_output = torch.cat([encoder_output[:, 0], encoder_output[:, img_embed.size(1)]], dim=1).view(B, 2 * 512)
        encoder_output = self.encoder_mlp(encoder_output)
        return encoder_output

    def compute_similarity(self, feat1, feat2):
        return torch.bmm(feat1.unsqueeze(1), feat2.unsqueeze(-1)).view(feat1.size(0),1)


    def get_cluster_feats(self, imgs, texts):
        outputs = []
        for img, text in zip(imgs, texts):
            outputs.append(self.shot_cls_fn(self.encode(img, text)))

        return outputs
    
    def get_order_score(self, input):
        img_tokens, text_tokens = input
        img0, img1 = img_tokens
        text0, text1 = text_tokens

        # embedding layer
        B = img0.size(0)

        # encoder
        enc0 = self.encode(img0, text0)
        enc1 = self.encode(img1, text1)

        # return torch.stack([encoder_output[:,pos0], encoder_output[:, pos1]], dim=0)

        value0 = self.value_fn(enc0)
        value1 = self.value_fn(enc1)

        return float(value1) - float(value0)

    
    def forward(self, input, gt=None):
        img_tokens, text_tokens = input

        img0, img1 = img_tokens[:2]
        text0, text1 = text_tokens[:2]
        negative_images = img_tokens[2:]
        negative_text = text_tokens[2:]


        # embedding layer
        B = img0.size(0)

        # encoder
        enc0 = self.encode(img0, text0)
        enc1 = self.encode(img1, text1)

        # return torch.stack([encoder_output[:,pos0], encoder_output[:, pos1]], dim=0)

        value0 = self.value_fn(enc0)
        value1 = self.value_fn(enc1)

        loss_temporal = 0.0
        if gt is not None:
            if gt["temp"] == 1:
                loss_temporal = - self.loss_temporal(value1 - value0)
            else:
                loss_temporal = - self.loss_temporal(value0 - value1)

        # print(loss_temporal.size())

        positive_score = torch.exp(self.compute_similarity(self.shot_cls_fn(enc0), self.shot_cls_fn(enc1)))

        if self.training:
            score_tot = positive_score.clone()
            for (img, txt) in zip(negative_images, negative_text):
                enc = self.encode(img, txt)
                score_tot += torch.exp(self.compute_similarity(self.shot_cls_fn(enc0), self.shot_cls_fn(enc)))

            loss_cl = - torch.log(torch.div(positive_score, score_tot))
            loss_shot = loss_temporal + self.lambda_cl * loss_cl

            return (value1 > value0).int().cpu(), loss_shot, loss_temporal, loss_cl

        else:
            num_shot_correct = 0
            for (img, txt) in zip(negative_images, negative_text):
                enc = self.encode(img, txt)
                score = torch.exp(self.compute_similarity(self.shot_cls_fn(enc0), self.shot_cls_fn(enc)))
                num_shot_correct += torch.gt(positive_score, score).int().cpu().item()
            loss_shot = loss_temporal

            return (value1 > value0).int().cpu(), loss_shot, num_shot_correct / len(negative_images)

    
