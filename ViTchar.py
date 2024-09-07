from re import L
import torch
from torch import nn, Tensor, optim
F = nn.functional
from torch.utils.data import Dataset, DataLoader
from bertTorchMaskless import BertSeq2Seq

from os.path import expanduser
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
import json

#Import CHAR
TrainPath = expanduser('~/Desktop/ML-Stuff/CharacterOCR-torch/CharTrain.npz')
TestPath = expanduser('~/Desktop/ML-Stuff/CharacterOCR-torch/CharTest.npz')
decoderJson = expanduser('~/Desktop/ML-Stuff/CharacterOCR-torch/TMNIST-Decoder.json')
train, test = np.load(TrainPath), np.load(TestPath)
Xtrain, Ytrain = Tensor(train.get('arr_0')), Tensor(train.get('arr_1'))
Xtest, Ytest = Tensor(test.get('arr_0')), Tensor(test.get('arr_1'))
num_to_label_map = json.load(open(decoderJson, 'r'))
# batch, 1, 28, 28 | batch, 94

class Dat(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

TrainDS = Dat(Xtrain, Ytrain)
TrainDL = DataLoader(TrainDS, batch_size=128, shuffle=True)
TestDS = Dat(Xtest[:10000], Ytest[:10000])
TestDL = DataLoader(TestDS, batch_size=1, shuffle=True)

class ViT_bert(nn.Module):
    def __init__(self, channels:int,ImageSize:int,PatchSize:int,numClasses:int,hiddedSize:int,feedforwardDim:int,numAttentionHeads:int,numLayers:int,attentionDropout:float,hiddenDropout:float):
        '''
        Constructs a Visual Transformer based on Bert
        Example Config for MNIST: channels=1,ImageSize=28,PatchSize=14,numClasses=94,hiddedSize=196,feedforwardDim=196,numAttentionHeads=14,numLayers=2,attentionDropout=0.1,hiddenDropout=0.1
        '''
        super().__init__()
        self.channels = channels
        self.image_size = ImageSize
        self.patch_size = PatchSize
        self.bert = BertSeq2Seq(hidden_size=hiddedSize,intermediate_size=feedforwardDim,num_attention_heads=numAttentionHeads,num_hidden_layers=numLayers,attention_probs_dropout_prob=attentionDropout,hidden_dropout_prob=hiddenDropout)
        self.patch_dim = self.channels * self.patch_size ** 2
        self.SeqLen = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)
        self.EmbedImg = nn.Conv2d(self.channels,self.patch_dim,self.patch_size,self.patch_size)
        self.norm = nn.LayerNorm(self.patch_dim)
        self.fc = nn.Linear((self.SeqLen*self.patch_dim),numClasses)

    def forward(self, img):
        Embed = self.EmbedImg(img).view(-1, self.SeqLen, self.patch_dim)
        Embed = self.norm(Embed)
        logits = self.bert(Embed)
        logits = self.norm(logits)
        logits = logits.flatten(1)
        logits = self.fc(logits)
        return logits

model = ViT_bert(channels=1,ImageSize=28,PatchSize=14,numClasses=94,hiddedSize=196,feedforwardDim=196,numAttentionHeads=14,numLayers=2,attentionDropout=0.1,hiddenDropout=0.1).to('mps')
opt = optim.AdamW(model.parameters(), 1e-4)

for i in range(EPOCHS:=1):
    model.train()
    for x,y in (t:=tqdm(TrainDL)):
        x = x.to('mps')
        y = y.to('mps')
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        t.set_description(f"loss: {loss.item():6.2f}")

    model.eval()
    with torch.no_grad():
        TotalCorrect = 0
        for x,y in (t:=tqdm(TestDL)):
            x = x.to('mps')
            y = y.to('mps')
            out = model(x).argmax()
            if out == y.argmax():
                TotalCorrect += 1
        print('Total Correct: ',TotalCorrect)
        print('Percent: ',TotalCorrect / len(TestDL) * 100)

StDict = model.state_dict()
torch.save(StDict,'ViTchar.pt')
