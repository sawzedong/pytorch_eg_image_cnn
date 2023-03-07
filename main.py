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

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--layers', type=int, default=3, help='number of CNN layers to use. accepts 1-4. default=3')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of CNN layers. default=3')
parser.add_argument('--savePath', default="model_cnnimage.pth", help='path to save model to')
parser.add_argument('--useFullDataset', action='store_true', help='whether to use full dataset or not (uses mini dataset instead)')
opt = parser.parse_args()

val_size = 250
batch_size = 32
num_epochs = opt.nEpochs
opt_func = torch.optim.Adam
lr = 0.001
dir_name = "full_input" if opt.useFullDataset else "input"

if __name__ == "__main__":
    # analysing dataset
    dataset_size = [0, 0, 0] # pred, test, train
    for dirname, _, filenames in os.walk(f'./{dir_name}'):
        for filename in filenames:
            if "pred" in dirname: dataset_size[0] += 1
            elif "test" in dirname: dataset_size[1] += 1
            elif "train" in dirname: dataset_size[2] += 1
    print("Dataset size: \nPrediction: {}\nTest: {}\nTrain: {}".format(dataset_size[0], dataset_size[1], dataset_size[2]))

    # loading dataset
    train_dir = f"./{dir_name}/seg_train/seg_train/"
    dataset = ImageFolder(train_dir,transform = transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))    
    train_size = len(dataset) - val_size 
    train_data, val_data = random_split(dataset,[train_size,val_size])
    train_dl = DeviceDataLoader(DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True), device)
    val_dl = DeviceDataLoader(DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True), device)

    model = to_device(NaturalSceneClassification(layers=opt.layers, kernel_size=opt.kernel_size),device)
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    test_dir = f"./{dir_name}/seg_test/seg_test/"
    test_dataset = ImageFolder(test_dir,transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
    print("After training:", evaluate(model, test_loader))

    torch.save(model.state_dict(), opt.savePath)