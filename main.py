import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from model import NaturalSceneClassification
from device_manager import get_default_device, to_device, DeviceDataLoader
from model_func import evaluate, fit
import argparse

torch.manual_seed(5)
device = get_default_device()

parser = argparse.ArgumentParser(description='PyTorch Image Recognition Example')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--layers', type=int, default=3, help='number of CNN layers to use. accepts 1-4. default=3')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of CNN layers. default=3')
parser.add_argument('--savePath', default="model_cnnimage.pth", help='path to save model to')
opt = parser.parse_args()

val_size = 2000
batch_size = 128
num_epochs = opt.nEpochs
opt_func = torch.optim.Adam
lr = 0.001

if __name__ == "__main__":
    # override protection
    if os.path.exists(opt.savePath):
        raise RuntimeError("File already exists at specified model save path")

    # analysing dataset
    dataset_size = [0, 0, 0] # pred, test, train
    for dirname, _, filenames in os.walk(f'./input'):
        for filename in filenames:
            if "pred" in dirname: dataset_size[0] += 1
            elif "test" in dirname: dataset_size[1] += 1
            elif "train" in dirname: dataset_size[2] += 1
    print("Dataset size: \nPrediction: {}\nTest: {}\nTrain: {}".format(dataset_size[0], dataset_size[1], dataset_size[2]))

    # loading dataset
    train_dir = f"./input/seg_train/seg_train/"
    dataset = ImageFolder(train_dir,transform = transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))    
    train_size = len(dataset) - val_size 
    train_data, val_data = random_split(dataset,[train_size,val_size])
    train_dl = DeviceDataLoader(DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True), device)
    val_dl = DeviceDataLoader(DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True), device)

    model = to_device(NaturalSceneClassification(layers=opt.layers, kernel_size=opt.kernel_size),device)
    model.layer_summary()
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    test_dir = f"./input/seg_test/seg_test/"
    test_dataset = ImageFolder(test_dir,transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
    print("After training:", evaluate(model, test_loader))

    torch.save(model.state_dict(), opt.savePath)