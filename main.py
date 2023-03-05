import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from model import NaturalSceneClassification
from device_manager import get_default_device, to_device, DeviceDataLoader
from model_func import evaluate, fit

val_size = 250
batch_size = 128
num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001

torch.manual_seed(5)
device = get_default_device()

if __name__ == "__main__":
    # analysing dataset
    dataset_size = [0, 0, 0] # pred, test, train
    for dirname, _, filenames in os.walk('./input'):
        for filename in filenames:
            if "pred" in dirname: dataset_size[0] += 1
            elif "test" in dirname: dataset_size[1] += 1
            elif "train" in dirname: dataset_size[2] += 1
    print("Dataset size: \nPrediction: {}\nTest: {}\nTrain: {}".format(dataset_size[0], dataset_size[1], dataset_size[2]))

    # loading dataset
    train_dir = "./input/seg_train/seg_train/"
    dataset = ImageFolder(train_dir,transform = transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))    
    train_size = len(dataset) - val_size 
    train_data, val_data = random_split(dataset,[train_size,val_size])
    train_dl = DeviceDataLoader(DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True), device)
    val_dl = DeviceDataLoader(DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True), device)

    model = to_device(NaturalSceneClassification(),device)
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    print("After training:", evaluate(model))

    torch.save(model.state_dict(), 'model_cnnimage.pth')