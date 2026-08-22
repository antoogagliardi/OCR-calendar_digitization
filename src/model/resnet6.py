# Libraries Import
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.classification import MulticlassAccuracy




class ResNet6(pl.LightningModule):
    def __init__(self, image_channels, num_classes, in_channels=16):
        super(ResNet6, self).__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(image_channels, in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # First block
        self.conv2= nn.Conv2d(self.in_channels, self.in_channels ,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(self.in_channels , self.in_channels , kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        # Identity Downsampling
        self.intermediate_channels = in_channels*2
        self.conv4 = nn.Conv2d(self.in_channels, self.intermediate_channels, kernel_size=1, stride=2, padding=0)
        self.bn4= nn.BatchNorm2d(self.intermediate_channels)

        # Second block
        self.conv5= nn.Conv2d(self.intermediate_channels, self.intermediate_channels ,kernel_size=3, stride=1, padding=1)
        self.bn5 = self.bn3= nn.BatchNorm2d(self.intermediate_channels)
        self.conv6= nn.Conv2d(self.intermediate_channels, self.intermediate_channels ,kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(self.intermediate_channels)
        # Average Pooling and Linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.intermediate_channels, num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        # First conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # First block  
        identity = x
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)

        # Identity Downsampling
        x= self.conv4(x)
        x = self.bn4(x)
        identity = x
        
        # Second block 
        x = self.bn5(self.conv5(x))
        x = self.relu(x)
        x = self.bn6(self.conv6(x))
        x += identity
        x = self.relu(x)

        # Average Pooling and Linear layer
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def on_train_epoch_start(self):
        self.t_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes,
                                        average="micro", ignore_index=None).to(self.device.type)
    
    def on_valid_epoch_start(self):
        self.v_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes,
                                        average="micro", ignore_index=None).to(self.device.type)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('train_loss', loss, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = self.v_acc(y_hat, y)
        
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return acc