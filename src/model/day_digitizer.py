from src.utils.plotting import plot_confusion_matrix
from src.model.resnet6 import ResNet6

import lightning.pytorch as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb


class DayDigitalizer(pl.LightningModule):
    def __init__(self, in_chan:int, out_dict:dict, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()        
        self.model = ResNet6(image_channels=in_chan, num_classes=len(out_dict), in_channels=16)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def on_train_epoch_start(self):
        self.t_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100).to(self.device.type)
        self.t_f1 = MulticlassF1Score(num_classes=len(self.hparams.out_dict), average="micro", ignore_index=None).to(self.device.type)
        self.t_cm = MulticlassConfusionMatrix(num_classes=len(self.hparams.out_dict), ignore_index=None).to(self.device.type)

    def on_validation_epoch_start(self):
        self.v_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100).to(self.device.type)
        self.v_f1 = MulticlassF1Score(num_classes=len(self.hparams.out_dict), average="micro", ignore_index=None).to(self.device.type)
        self.v_cm = MulticlassConfusionMatrix(num_classes=len(self.hparams.out_dict), ignore_index=None).to(self.device.type)

    def training_step(self, batch, batch_idx):
        image = batch["image"].pixel_values
        groud_label = batch["label"]

        logits = self.model(image)
        logits = F.softmax(logits, dim=-1)

        # Loss Computation
        loss = self.t_loss(logits, groud_label)

        # Accuracy Computation
        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            f1_score = self.t_f1(pred, groud_label)
            self.t_cm.update(pred, groud_label)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "train_f1": f1_score}
    
    def validation_step(self, batch, batch_idx):
        image = batch["image"].pixel_values
        groud_label = batch["label"]

        logits = self.model(image)
        # logits = F.softmax(logits, dim=-1)
        
        # Loss Computation
        loss = self.v_loss(logits, groud_label)

        # Accuracy Computation
        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            f1_score = self.v_f1(pred, groud_label)
            self.v_cm.update(pred, groud_label)
        
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid_f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)

        return {"valid_loss": loss, "valid_f1": f1_score}
    
    def on_train_epoch_end(self):
        print("Train F1 Score: ", self.t_f1.compute().item())
        cm_path = os.path.join(os.getcwd(), f"cm/{wandb.run.name}_{wandb.run.id}")
        # Log the Validation Confusion Matrix Image into WandB
        print(f"-- Logging Training Confusion Matrix of the epoch {self.current_epoch} --")
        cm = self.t_cm.compute().detach().cpu().numpy()
        plot_confusion_matrix(cm=cm, out_class=list(self.hparams.out_dict.values()),
                              cmap="Blues", dim=(20,20), title=f"Training Confusion Matrix Epoch {self.current_epoch}",
                              save_cm=True, save_dir=cm_path, file_name=f"train_cm_{self.current_epoch}")
        wandb.log({"Training Confusion Matrices": wandb.Image(os.path.join(cm_path, f"train_cm_{self.current_epoch}.png"))})
        print("-- Confusion Matrix saved --")
        self.t_cm.reset()

        print("Valid F1 Score: ", self.v_f1.compute().item())
        cm_path = os.path.join(os.getcwd(), f"cm/{wandb.run.name}_{wandb.run.id}")
        # Log the Validation Confusion Matrix Image into WandB
        print(f"-- Logging Validation Confusion Matrix of the epoch {self.current_epoch} --")
        cm = self.v_cm.compute().detach().cpu().numpy()
        plot_confusion_matrix(cm=cm, out_class=list(self.hparams.out_dict.values()),
                              cmap="Reds", dim=(20,20), title=f"Validation Confusion Matrix Epoch {self.current_epoch}",
                              save_cm=True, save_dir=cm_path, file_name=f"valid_cm_{self.current_epoch}")
        wandb.log({"Validation Confusion Matrices": wandb.Image(os.path.join(cm_path, f"valid_cm_{self.current_epoch}.png"))})
        print("-- Confusion Matrix saved --")
        self.v_cm.reset()