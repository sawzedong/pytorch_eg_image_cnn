import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self, layers=3, kernel_size=3):
        super().__init__()
        k = kernel_size
        w = 1
        if(k == 5): w = 2
        elif (k == 7): w=3 
        self.network = 0
        if layers == 1:
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size = k, padding = w), nn.ReLU(), nn.Conv2d(32,64, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2)
            )
        elif layers == 2:
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size = k, padding = w), nn.ReLU(), nn.Conv2d(32,64, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.Conv2d(128 ,128, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2)
            )
        elif layers == 4:
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size = k, padding = w), nn.ReLU(), nn.Conv2d(32,64, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.Conv2d(128 ,128, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(128, 256, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.Conv2d(256,256, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(256, 512, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.Conv2d(512,512, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2)
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size = k, padding = w), nn.ReLU(), nn.Conv2d(32,64, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.Conv2d(128 ,128, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(128, 256, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.Conv2d(256,256, kernel_size = k, stride = w, padding = w), nn.ReLU(), nn.MaxPool2d(2,2)
            )
        matrix_size = self.network(torch.rand(32, 3, 150, 150)).shape
        matrix_columns = 1
        for j in matrix_size: matrix_columns*=j
        matrix_columns = matrix_columns // 32
        self.network.append(nn.Flatten())
        self.network.append(nn.Linear(matrix_columns,1024))
        self.network.append(nn.ReLU())
        self.network.append(nn.Linear(1024,512))
        self.network.append(nn.ReLU())
        self.network.append(nn.Linear(512,6))
    def layer_summary(self):
        summary(self.network, (3, 150, 150))
    def forward(self, xb):
        return self.network(xb)