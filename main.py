import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from model import NaturalSceneClassification
from device_manager import get_default_device, to_device, DeviceDataLoader
from PIL import Image

val_size = 2000
batch_size = 128
num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001

@torch.no_grad()
def evaluate(model, val_loader):
    # evaluate model performance
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    # fit model to training data
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def plot_losses(history):
    # plot losses in each epoch
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def predict_img_class(img,model):
    # Predict the class of image and Return Predicted Class
    img = to_device(img.unsqueeze(0), device)
    prediction =  model(img)
    _, preds = torch.max(prediction, dim = 1)
    return dataset.classes[preds[0].item()]

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
    test_dir = "./input/seg_test/seg_test/"

    dataset = ImageFolder(train_dir,transform = transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    test_dataset = ImageFolder(test_dir,transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))

    random_seed = 2021
    torch.manual_seed(random_seed)
    
    train_size = len(dataset) - val_size 
    train_data,val_data = random_split(dataset,[train_size,val_size])

    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    model = to_device(NaturalSceneClassification(),device)

    print("Before training:", evaluate(model, val_dl))

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    plot_losses(history)

    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
    print("After training:", evaluate(model, test_loader))

    torch.save(model.state_dict(), 'natural-scene-classification.pth')

    # select image to predice
    img = Image.open("../input/intel-image-classification/seg_pred/seg_pred/10004.jpg")
    img = transforms.ToTensor()(img)
    plt.imshow(img.permute(1,2,0))
    print(f"Predicted Class : {predict_img_class(img,model)}")