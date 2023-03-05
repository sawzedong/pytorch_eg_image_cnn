import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from device_manager import get_default_device, to_device, DeviceDataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

device = get_default_device()
batch_size = 128

@torch.no_grad()
def evaluate(model):
    # evaluate model performance
    test_dir = "./input/seg_test/seg_test/"
    test_dataset = ImageFolder(test_dir,transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
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

def predict_img_class(img, model):
    # Predict the class of image and Return Predicted Class
    img = to_device(img.unsqueeze(0), device)
    prediction =  model(img)
    _, preds = torch.max(prediction, dim = 1)
    train_dir = "./input/seg_train/seg_train/"
    dataset = ImageFolder(train_dir,transform = transforms.Compose([
        transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    return dataset.classes[preds[0].item()]
