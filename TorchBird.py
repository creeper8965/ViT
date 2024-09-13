from TorchVIT import ViT_bert
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.transforms import v2
from os.path import expanduser
import tqdm
import wandb
import itertools

IMG_SIZE = 224
BATCH = 16
LR = 1e-5
EPOCHS = 10
DEVICE = torch.device('mps')

def LoadData():
    dataPath = expanduser('~/Desktop/ML-Stuff/Tdata/BirdSet/BirdTrain')
    valPath = expanduser('~/Desktop/ML-Stuff/Tdata/BirdSet/BirdTest')
    transforms = v2.Compose([v2.ToImage(),v2.Resize((IMG_SIZE,IMG_SIZE)),v2.ToDtype(torch.float32, scale=True),v2.RandomAutocontrast(),v2.RandomHorizontalFlip()]) #,v2.GaussianBlur(kernel_size=(23,23))
    ds = datasets.ImageFolder(dataPath, transforms)
    dl = DataLoader(dataset=ds,batch_size=BATCH,shuffle=True, pin_memory=False, num_workers=4)
    testDS = datasets.ImageFolder(valPath, transforms)
    testDL = DataLoader(dataset=testDS,batch_size=BATCH,shuffle=True, pin_memory=False, num_workers=4)
    ttImgs, NumClasses = len(ds.imgs), len(ds.classes)
    return dl, ttImgs, NumClasses, testDL

def ValStep(testDL, model):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        img, labels = next(testDL)
        img, labels = img.to(DEVICE), labels.to(DEVICE)
        out = model(img)
        loss = lossfn(out, labels)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    model.train()
    return correct / total  # This gives you accuracy


if __name__ == '__main__':
    wandb.init(
        # set the wandb project where this run will be logged
        project="ViT",

        # track hyperparameters and run metadata
        config={
        "learning_rate": LR,
        "architecture": "ViT",
        "dataset": "birbs",
        "epochs": EPOCHS,
        }
    )

    dl, ttImgs, NumClasses, testDL = LoadData()
    testDL = itertools.cycle(iter(testDL))
    print(f'Number of Classes;{NumClasses}')
    model = ViT_bert(channels=3, ImageSize=224, PatchSize=28, numClasses=NumClasses, hiddenSize=384, feedforwardDim=768, numAttentionHeads=12, numLayers=4, attentionDropout=0.1, hiddenDropout=0.1)
    model.to(DEVICE)
    optim = optim.AdamW(model.parameters(), LR)
    lossfn = nn.functional.cross_entropy

    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        with tqdm.tqdm(dl, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch') as t:
            for batch_idx, (inputs, labels) in enumerate(t):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optim.zero_grad()
                outputs = model.forward(inputs)
                loss = lossfn(outputs, labels)
                loss.backward()
                optim.step()
                lossIt = loss.item()
                running_loss += lossIt * inputs.size(0)
                acc = ValStep(testDL, model)
                wandb.log({"acc": acc, "loss": lossIt})
                t.set_postfix(loss=lossIt)
                t.update()

            epoch_loss = running_loss / ttImgs
            print(f'Epoch [{epoch+1}/{EPOCHS}], Total Loss: {epoch_loss:.4f}')
    wandb.finish()
    model.eval()
    example = torch.rand((1,3,224,224),dtype=torch.float32).to('mps')
    # onnx_program = torch.onnx.dynamo_export(model, example)
    # onnx_program.save('ViTgenderclassifier.onnx')
    tModel = torch.jit.trace(model, example)
    torch.save(tModel, 'ViTbirdTraced.pt')
    import coremltools as ct
    model_from_torch = ct.convert(tModel,convert_to="mlprogram",inputs=[ct.TensorType(name="input", shape=example.shape)], compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16)
    model_from_torch.save('Vitbirdspecie.mlpackage')
