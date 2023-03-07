import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from model import NaturalSceneClassification
from device_manager import get_default_device, to_device, DeviceDataLoader
from model_func import evaluate
import argparse

device = get_default_device()

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--modelPath', default="model_cnnimage.pth", help='path to load model frmo')
parser.add_argument('--useFullDataset', action='store_true', help='whether to use full dataset or not (uses mini dataset instead)')
parser.add_argument('--layers', type=int, default=3, help='number of CNN layers to use. accepts 1-4. default=3')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of CNN layers. default=3')
opt = parser.parse_args()
dir_name = "full_input" if opt.useFullDataset else "input"

batch_size = 32
if __name__ == "__main__":
    model = to_device(NaturalSceneClassification(layers=opt.layers, kernel_size=opt.kernel_size),device)
    model.load_state_dict(torch.load(opt.modelPath, map_location=device))

    test_dir = f"./{dir_name}/seg_test/seg_test/"
    test_dataset = ImageFolder(test_dir,transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
    
    print("After training:", evaluate(model, test_loader))
    