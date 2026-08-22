# Libraries Import
from src.model.unet import UNet
from src.utils.dice_loss import SoftDiceLoss
from src.utils.plotting import plot_confusion_matrix
from src.utils.utility import retrieve_relevant_px_value

import lightning.pytorch as pl
import torch
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import wandb
import os




class CalendarParser(pl.LightningModule):
    def __init__(self, n_channels, bilinear=False):
        super(CalendarParser, self).__init__()
        self.class_to_color = {0: 0, 1: 25, 2: 50, 3: 75, 4: 90, 5: 115}
        self.label_to_color = {"none(contour)": 0,
                               "month": 25,
                               "day": 50,
                               "note": 75,
                               "emptyNote": 90,
                               "border": 115}
        self.color_to_label = {}
        for key, val in self.label_to_color.items(): self.color_to_label[val] = key
        
        # Model
        self.model = UNet(n_channels=n_channels, n_classes=len(self.label_to_color), bilinear=bilinear)

        # Metrics
        self.loss = SoftDiceLoss(reduction="mean", use_softmax=True)

    def forward(self, x):
        output = self.model(x)

        return output
        
    def configure_optimizers(self):
        learning_rate = 0.0001
        optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        return optim
    
    def on_train_epoch_start(self):
        self.t_f1_score = MulticlassF1Score(num_classes=len(self.label_to_color.keys()),
                                            average="micro", ignore_index=None).to(self.device.type)
        self.t_cm = MulticlassConfusionMatrix(num_classes=len(self.label_to_color.keys()),
                                              normalize="all", ignore_index=None).to(self.device.type)
    
    def on_validation_epoch_start(self):
        self.v_f1_score = MulticlassF1Score(num_classes=len(self.label_to_color.keys()),
                                            average="micro", ignore_index=None).to(self.device.type)
        self.v_cm = MulticlassConfusionMatrix(num_classes=len(self.label_to_color.keys()),
                                              normalize="all", ignore_index=None).to(self.device.type)
    
    def training_step(self, batch, batch_idx):
        img_in = batch["image"]                             # ; print("Image shape: ", img_in.shape)
        ground_truth = batch["ground_mask"].squeeze()       # ; print("Mask shape: ", ground_truth.shape)
        
        logits = self(img_in)                               # ; print("Logits shape: ", ground_truth.shape)

        # Loss Function
        loss = self.loss(logits, ground_truth)
        
        # Accuracy Evaluation
        with torch.no_grad():
            out = torch.argmax(logits, dim=1)
            f1_score = self.t_f1_score(torch.flatten(out), torch.flatten(ground_truth))
            self.t_cm.update(out.view(out.shape[0], -1), ground_truth.view(ground_truth.shape[0], -1))
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)
        wandb.log({"train_loss": loss})

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        img_in = batch["image"]                             #; print("Image shape: ", img_in.shape)
        ground_truth = batch["ground_mask"].squeeze()       #; print("Mask shape: ", ground_truth.shape)
        
        logits = self(img_in)                               #; print("Logits shape: ", ground_truth.shape)
        
        # Loss Function
        loss = self.loss(logits, ground_truth)

        # Accuracy Evaluation
        with torch.no_grad():
            out = torch.argmax(logits, dim=1)
            f1_score = self.v_f1_score(torch.flatten(out), torch.flatten(ground_truth))
            self.v_cm.update(out.view(out.shape[0], -1), ground_truth.view(ground_truth.shape[0], -1))

        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid_f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)

        return {"valid_loss": loss}
    
    def on_train_epoch_end(self):
        print("Train F1 Score: ", self.t_f1_score.compute().item())
        cm_path = os.path.join(os.getcwd(), f"cm/{wandb.run.name}_{wandb.run.id}")
        # Log the Validation Confusion Matrix Image into WandB
        print(f"-- Logging Training Confusion Matrix of the epoch {self.current_epoch} --")
        cm = self.t_cm.compute().detach().cpu().numpy()
        plot_confusion_matrix(cm=cm, out_class=list(self.label_to_color.keys()),
                              cmap="Blues", dim=(5,10), title=f"Training Confusion Matrix Epoch {self.current_epoch}",
                              save_cm=True, save_dir=cm_path, file_name=f"train_cm_{self.current_epoch}")
        wandb.log({"Training Confusion Matrices": wandb.Image(os.path.join(cm_path, f"train_cm_{self.current_epoch}.png"))})
        print("-- Confusion Matrix saved --")
        self.t_cm.reset()
  
        print("Valid F1 Score: ", self.v_f1_score.compute().item())
        cm_path = os.path.join(os.getcwd(), f"cm/{wandb.run.name}_{wandb.run.id}")
        # Log the Validation Confusion Matrix Image into WandB
        print(f"-- Logging Validation Confusion Matrix of the epoch {self.current_epoch} --")
        cm = self.v_cm.compute().detach().cpu().numpy()
        plot_confusion_matrix(cm=cm, out_class=list(self.label_to_color.keys()),
                              cmap="Reds", dim=(5,10), title=f"Validation Confusion Matrix Epoch {self.current_epoch}",
                              save_cm=True, save_dir=cm_path, file_name=f"valid_cm_{self.current_epoch}")
        wandb.log({"Validation Confusion Matrices": wandb.Image(os.path.join(cm_path, f"valid_cm_{self.current_epoch}.png"))})
        print("-- Confusion Matrix saved --")
        self.v_cm.reset()

    def predict(self, device:torch.device, data:torch.Tensor, ground_comparison:bool=False):        
        if ground_comparison == True:   logits = self(data["image"].to(device))
        else:   logits = self(data.to(device))
        print("Logits Shape: ", logits.shape)
        out = torch.argmax(logits, dim=1)
        print("Prediction output: ", out.shape)

        # Map values inside the out tensor with color class values using advanced indexing
        # value_tensor = torch.tensor(list(model.class_to_color.values())).to(DEVICE)
        mapped_out = out # value_tensor[out].to(dtype=torch.float32)
        print(retrieve_relevant_px_value(mapped_out.cpu().detach().numpy()))
        
        return mapped_out