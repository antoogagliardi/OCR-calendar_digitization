# Libraries Import
from src.model.unet import UNet
from src.utils.dice_loss import SoftDiceLoss

import lightning.pytorch as pl
import torch
import wandb










class WordParser(pl.LightningModule):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(WordParser, self).__init__()
        self.label_to_color = {"none": 0,
                               "word": 25}
        self.color_to_label = {0: "none",
                               255: "word"}
        self.class_to_color = {0: 0, 1: 255}

        # Model
        self.model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)

        # Metrics
        self.loss = SoftDiceLoss(reduction="mean", use_softmax=True)
        self.acc = 0

    def forward(self, x):
        output = self.model(x)

        return output
        
    def configure_optimizers(self):
        learning_rate = 0.0001
        optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        return optim
    
    def on_validation_epoch_start(self):
        self.batches = 0
    
    def training_step(self, batch, batch_idx):
        img_in = batch["image"].pixel_values                #; print("Image shape: ", img_in.shape)# ; pprint(img_in)
        ground_truth = batch["ground_mask"].squeeze()       #; print("Mask shape: ", ground_truth.shape)# ; pprint(ground_truth)
        
        logits = self(img_in)                               #; print("Logits shape: ", ground_truth.shape)# ; pprint(ground_truth)

        # Loss Function
        loss = self.loss(logits, ground_truth)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        wandb.log({"train_loss": loss})

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        img_in = batch["image"].pixel_values                #; print("Image shape: ", img_in.shape)#; pprint(img_in)
        ground_truth = batch["ground_mask"].squeeze()       #; print("Mask shape: ", ground_truth.shape)#; pprint(ground_truth)
        
        logits = self(img_in)                               #; print("Logits shape: ", ground_truth.shape)#; pprint(ground_truth)
        
        # Loss Function
        valid_loss = self.loss(logits, ground_truth)

        # Accuracy Evaluation
        with torch.no_grad():
            out = torch.argmax(logits, dim=1)
            mask_sim = torch.cosine_similarity(out.view(out.shape[0], -1),
                                               ground_truth.view(ground_truth.shape[0], -1))
            mask_sim = torch.mean(mask_sim, dim=0)
            self.acc += mask_sim.item()
            self.batches += 1

        
        self.log("valid_loss", valid_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"valid_loss": valid_loss}
    
    def on_train_epoch_end(self):
        print("Total Batches: ", self.batches)
        print("Cumulated Accuracy: ", self.acc)

        self.acc = self.acc / self.batches
        
        self.log("valid_f1", self.acc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.batches = 0
        self.acc = 0