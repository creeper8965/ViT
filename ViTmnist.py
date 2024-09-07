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

#Import MNIST
TestDF = np.load(expanduser('~/Desktop/ML-Stuff/Tdata/MnistTest.npz'))
TrainDF = np.load(expanduser('~/Desktop/ML-Stuff/Tdata/MnistTrain.npz'))

Xtrain, Ytrain = Tensor(TrainDF['arr_0']), Tensor(TrainDF['arr_1'])
Xtest, Ytest = Tensor(TestDF['arr_0']), Tensor(TestDF['arr_1'])
#60000,28,28 - 60000,
Xtrain = Xtrain.unsqueeze(1)
Xtest = Xtest.unsqueeze(1)
#60000,1,28,28 - 60000,

class Dat(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ViT_bert(nn.Module):
    def __init__(self, channels,ImageSize,PatchSize,num_classes):
        super().__init__()
        self.channels = channels
        self.image_size = ImageSize
        self.patch_size = PatchSize
        self.bert = BertSeq2Seq(hidden_size=196,intermediate_size=320,num_attention_heads=7,num_hidden_layers=4,attention_probs_dropout_prob=0.1,hidden_dropout_prob=0.1)
        self.patch_dim = self.channels * self.patch_size ** 2
        self.SeqLen = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)
        self.EmbedImg = nn.Conv2d(self.channels,self.patch_dim,self.patch_size,self.patch_size)
        self.norm = nn.LayerNorm(self.patch_dim)
        self.fc = nn.Linear(784,num_classes)

    def forward(self, img):
        Embed = self.EmbedImg(img).view(-1, self.SeqLen, self.patch_dim)
        Embed = self.norm(Embed)
        logits = self.bert(Embed).flatten(1)
        logits = self.fc(logits)
        return logits

model = ViT_bert(1,28,14,10).to('mps')
opt = optim.AdamW(model.parameters(), 1e-4)

TrainDS = Dat(Xtrain, Ytrain)
TrainDL = DataLoader(TrainDS, batch_size=64, shuffle=True)
TestDS = Dat(Xtest[:1000], Ytest[:1000])
TestDL = DataLoader(TestDS, batch_size=1, shuffle=True)

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
            if out == y:
                TotalCorrect += 1
        print('Total Correct: ',TotalCorrect)
        print('Percent: ',TotalCorrect / len(TestDL) * 100)

StDict = model.state_dict()
torch.save(StDict,'ViTmnist.pt')
