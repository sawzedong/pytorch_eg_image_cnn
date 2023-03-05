import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from model import NaturalSceneClassification
from device_manager import get_default_device, to_device, DeviceDataLoader
from model_func import evaluate

device = get_default_device()

batch_size = 32
if __name__ == "__main__":
    model = to_device(NaturalSceneClassification(),device)
    model.load_state_dict(torch.load('./model_saves/cnnimage_200epoch_3convlayer.pth', map_location=device))

    test_dir = "./input/seg_test/seg_test/"
    test_dataset = ImageFolder(test_dir,transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
    
    print("After training:", evaluate(model, test_loader))
    