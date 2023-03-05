import torch
from model import NaturalSceneClassification
from device_manager import get_default_device, to_device
from model_func import evaluate

device = get_default_device()

if __name__ == "__main__":
    model = to_device(NaturalSceneClassification(),device)
    model.load_state_dict(torch.load('./model_saves/cnnimage_200epoch_3convlayer.pth', map_location=device))
    
    print("After training:", evaluate(model))
    