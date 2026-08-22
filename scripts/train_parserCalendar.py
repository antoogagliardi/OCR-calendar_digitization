import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
from pprint import pprint
import os
import yaml
import random
import collections
import wandb

#-- Pytorch
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.data.calendardb import CalendarDataset
from src.data.calendarloader import CalendarDataModule
from src.model.calendar_parser import CalendarParser

print("Libraries imported")


# Read configuration file
with open("configs/config_parserCalendar.yaml") as f:
    cfg = yaml.safe_load(f)
print("== Configuration File ==")
pprint(cfg)

# Setup device
DEVICE = torch.device(cfg["project"]["device"])
print(f"- Device used: {DEVICE}")

## Per riprodurre su un altro host è sufficiente modificare il root path con quello
##  in cui sono attualmente contenute le cartelle "code" e "dataset". Il resto del codice
##  si adatterà di conseguenza.
root_path = os.getcwd()
data_path = os.path.join(root_path, cfg["paths"]["data"])
model_data_path = os.path.join(root_path, "model_data")
model_ckpt_path = os.path.join(root_path, "ckpt")
print("root: ", root_path)
print("datasets: ", data_path)
print("model ckpt: ", model_ckpt_path)


# Dataset Creation and Data Batching
# Define a dictionary of color to plot the ground truth mask
calendar_color_dict = {0: [0, 0, 0],        # none
                       1: [0, 255, 0],      # month 
                       2: [0, 0, 255],      # day 
                       3: [128, 128, 128],  # note 
                       4: [255, 128, 0],    # emptyNote 
                       5: [255, 0, 0] }     # border

SPLIT_FOL   = cfg["paths"]["data_split"]
RELOAD_DATA = cfg["training"]["reload_data"]
BATCH_SIZE  = cfg["training"]["batch_size"]


if RELOAD_DATA == True:
    calendar_data_manager = CalendarDataModule(dataset=None, test_set=None,
                                               saved_db_folder=SPLIT_FOL,
                                               batch_size=BATCH_SIZE, reload_data=True, drop_last=True, device=DEVICE)
else:
    calendar_dataset = CalendarDataset(data_path=data_path, mode="BW", downscale=True)
    print("Length of Calendar Dataset: ", len(calendar_dataset))

    # Example of input data
    for idx in range(len(calendar_dataset)):
        if "a1" in calendar_dataset[idx]["id"]:
            print("ID: ", calendar_dataset[idx]["id"])
            print("MONTH: ", calendar_dataset[idx]["month"])
            print("CALENDAR IMAGE SHAPE: ", calendar_dataset[idx]["image"].shape)
            print("CALENDAR MASK SHAPE: ", calendar_dataset[idx]["label"].shape)
            fig, ax = plt.subplots(1, 2, figsize=(10,5))
            ax[0].imshow(calendar_dataset[idx]["image"][0], cmap="gray")    ; ax[0].set_title("Image")
            ax[1].imshow(calendar_dataset[idx]["label"][0], cmap="viridis") ; ax[1].set_title("Mask")
            plt.tight_layout()
            plt.show()

    calendar_color_dict = {
        0: {"name": "Black", "rgb": [0, 0, 0]},                         # none
        1: {"name": "Dark Yellow", "rgb": [200, 200, 0]},               # month
        2: {"name": "Plum 4", "rgb": [101, 27, 139]},                   # day
        3: {"name": "Dark Magenta", "rgb": [200, 0, 200]},              # note
        4: {"name": "Dark Cyan", "rgb": [0, 200, 200]},                 # emptyNote
        5: {"name": "Medium Grey", "rgb": [150, 150, 150]},             # border
    }
    mask = []
    idx = random.randint(0, 20)
    for i in range(calendar_dataset[idx]["label"].shape[1]):
        row = []
        for j in range(calendar_dataset[idx]["label"].shape[2]):
            row.append(calendar_color_dict[calendar_dataset[idx]["label"][0][i,j]]["rgb"])
        mask.append(row)
    mask = np.array(mask, dtype=np.uint)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.imshow(mask)
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(calendar_dataset[0]["image"][0], cmap="gray"); ax[0].set_title("Image")
    ax[1].imshow(calendar_dataset[0]["label"][0], cmap="gray"); ax[1].set_title("Mask")
    plt.tight_layout()
    plt.show()


    calendar_data_manager = CalendarDataModule(dataset=calendar_dataset, test_set=None,
                                               saved_db_folder=SPLIT_FOL,
                                               batch_size=1, reload_data=False, drop_last=True, device=DEVICE)



next(iter(calendar_data_manager.train_dataloader()))
t_freq = collections.Counter()
v_freq = collections.Counter()
for i in range(len(calendar_data_manager.train_dataset)): t_freq[calendar_data_manager.train_dataset[i]["month"]] += 1
for i in range(len(calendar_data_manager.valid_dataset)): v_freq[calendar_data_manager.valid_dataset[i]["month"]] += 1
print("Train Data Freq: ", t_freq)
print("Valid Data Fred: ", v_freq)






# Model Training
RESUME_TRAINING = cfg["training"]["resume"]
GRADIENT_ACCUMULATION = cfg["training"]["grad_accumulation"]
CKPT_PATH = cfg["training"]["ckpt_path"]
PROJECT_NAME = cfg["training"]["wandb_proj"]
RUN_NAME = cfg["training"]["wandb_run"]
# LAST_RUN_ID is discovered once the project is created

config = {"batch_size": BATCH_SIZE,
          "learning_rate": 0.0001,
          "gradient_accumulation": ("YES", GRADIENT_ACCUMULATION) if GRADIENT_ACCUMULATION > 1 else ("NO", None)}

if RESUME_TRAINING == True:
    LAST_RUN_ID = cfg["training"]["wandb_runID"]
    run = wandb.init(project=PROJECT_NAME, name=RUN_NAME,
                     config=config,
                     resume=True, id=LAST_RUN_ID)
    LAST_EPOCHS = cfg["training"]["last_epoch"]
    CKPT = os.path.join(model_ckpt_path, f"{RUN_NAME}_{LAST_RUN_ID}/model-epoch={LAST_EPOCHS - 1}.ckpt")
    ADDITTIONAL_EPOCHS = cfg["training"]["add_epoch"]

    EPOCHS = LAST_EPOCHS + ADDITTIONAL_EPOCHS

else:
    run = wandb.init(project=PROJECT_NAME, name=RUN_NAME, config=config)
    EPOCHS = cfg["training"]["epochs"]

# Note: MPS Backend doesn't support torch.DoubleTensor(=float64)
if DEVICE.type == "mps":    model = CalendarParser(n_channels=1, bilinear=False).to(device=DEVICE, dtype=torch.float32)
else:                       model = CalendarParser(n_channels=1, bilinear=False).to(device=DEVICE)
print(model)

wandb_logger = WandbLogger(name=RUN_NAME,
                           save_dir=CKPT_PATH, offline=False,
                           project=PROJECT_NAME, log_model=False) # log_model = "all"/True/False

trainer = pl.Trainer(accelerator=DEVICE.type,   # gpu, cpu, mps
                     num_sanity_val_steps=2,
                     accumulate_grad_batches=GRADIENT_ACCUMULATION,
                     logger=wandb_logger,
                     devices=1, max_epochs=EPOCHS,
                     callbacks=[EarlyStopping(monitor="valid_loss", min_delta=0.001, patience=5, mode="min"),
                                EarlyStopping(monitor="valid_f1", min_delta=0.001, patience=9, mode="max"),
                                ModelCheckpoint(dirpath=f"{CKPT_PATH}/{RUN_NAME}_{wandb.run.id}",
                                                filename='model-{epoch}',
                                                save_top_k=-1)],
                     log_every_n_steps=1)

if RESUME_TRAINING == True:
    trainer.fit(model, datamodule=calendar_data_manager, ckpt_path=CKPT)
else:
    trainer.fit(model, datamodule=calendar_data_manager)

print("Training Complete")
wandb.finish()
print("End of the training program")