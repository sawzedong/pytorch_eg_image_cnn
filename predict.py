import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import NaturalSceneClassification
from device_manager import get_default_device, to_device
from PIL import Image
from model_func import predict_img_class
import argparse

device = get_default_device()

parser = argparse.ArgumentParser(description='PyTorch Image Recognition Example')
parser.add_argument('--modelPath', default="model_cnnimage.pth", help='path to load model from')
parser.add_argument('--layers', type=int, default=3, help='number of CNN layers to use. accepts 1-4. default=3')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of CNN layers. default=3')
parser.add_argument('--imgPath', default="./input/seg_pred/seg_pred/591.jpg", help='path to load image from')
opt = parser.parse_args()

train_dir = "./input/seg_train/seg_train/"
dataset = ImageFolder(train_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

if __name__ == "__main__":
    model = to_device(NaturalSceneClassification(layers=opt.layers, kernel_size=opt.kernel_size),device)
    model.load_state_dict(torch.load(opt.modelPath, map_location=device))
    model.layer_summary()

    # select image to predict
    img = Image.open(opt.imgPath)
    img = transforms.ToTensor()(img)
    plt.imshow(img.permute(1,2,0))
    print(f"Predicted Class : {predict_img_class(img,model)}")