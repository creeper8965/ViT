import torch
from torch import nn, Tensor, optim
F = nn.functional
from torch.utils.data import Dataset, DataLoader
from TorchViT import ViT_bertCLS

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

model = ViT_bertCLS(channels=1,ImageSize=28,PatchSize=14,numClasses=10,hiddenSize=196,feedforwardDim=320,numAttentionHeads=7,numLayers=4,attentionDropout=0.1,hiddenDropout=0.1).to('mps')
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
