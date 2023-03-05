import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import NaturalSceneClassification
from device_manager import get_default_device, to_device
from PIL import Image
from model_func import predict_img_class

device = get_default_device()

train_dir = "./input/seg_train/seg_train/"
dataset = ImageFolder(train_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

if __name__ == "__main__":
    model = to_device(NaturalSceneClassification(),device)
    model.load_state_dict(torch.load('./model_saves/cnnimage_200epoch_3convlayer.pth', map_location=device))

    # select image to predict
    img = Image.open("./input/seg_pred/seg_pred/591.jpg")
    img = transforms.ToTensor()(img)
    plt.imshow(img.permute(1,2,0))
    print(f"Predicted Class : {predict_img_class(img,model)}")