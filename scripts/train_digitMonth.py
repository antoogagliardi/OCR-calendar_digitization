import yaml
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb
from pprint import pprint


from src.data.monthdb import MonthDataset
from src.data.monthloader import MonthDataModule
from src.model.month_digitizer import MonthDigitalizer


print("Libraries imported")


# Read configuration file
with open("configs/config_digitMonth.yaml") as f:
    cfg = yaml.safe_load(f)
print("== Configuration File ==")
pprint(cfg)

# Setup device
DEVICE = torch.device(cfg["project"]["device"])
print(f"- Device used: {DEVICE}")


# Paths
root_path = os.getcwd()
data_path = os.path.join(root_path, cfg["paths"]["data"])

model_data_path = os.path.join(root_path, "model_data")

model_ckpt_path = os.path.join(root_path, "ckpt")

print("root: ", root_path)
print("datasets: ", data_path)
print("model ckpt: ", model_ckpt_path)



## Create Months Vocabulary
RELOAD_DICT = cfg["training"]["reload_dict"]

if RELOAD_DICT == True:
    idx_to_month = torch.load(os.path.join(model_data_path, f'month_dictionary.dict'))["idx_month"]
    month_to_idx = torch.load(os.path.join(model_data_path, f'month_dictionary.dict'))["month_idx"]
    print("length of model vocabulary: ", len(idx_to_month))
    print(idx_to_month)
else:    
    month_to_idx = {"january": 0,
                    "february": 1, 
                    "march": 2, 
                    "april": 3, 
                    "may": 4, 
                    "june": 5, 
                    "july": 6, 
                    "august": 7, 
                    "september": 8,
                    "october": 9, 
                    "november": 10, 
                    "december": 11}
    idx_to_month = {}
    for key, val in month_to_idx.items():
        idx_to_month[val] = key
    print("length of model vocabulary: ", len(idx_to_month))
    pprint(idx_to_month, sort_dicts=True)
    
    torch.save({"idx_month": idx_to_month,
                "month_idx": month_to_idx}, os.path.join(model_data_path, f'month_dictionary.dict'))


# Dataset Creation and Data Batching
SPLIT_FOL   = cfg["paths"]["data_split"]
RELOAD_DATA = cfg["training"]["reload_data"]
BATCH_SIZE = cfg["training"]["batch_size"]

if RELOAD_DATA == True:
    months_data_manager = MonthDataModule(encode_dict=month_to_idx, dataset=None, test_set=None, 
                                          saved_db_folder=SPLIT_FOL, 
                                          batch_size=BATCH_SIZE, reload_data=True, drop_last=True)
else:
    month_train_dataset = MonthDataset(data_path=data_path,
                                       df=pd.read_csv(os.path.join(data_path, "month.csv")),
                                       augment_step=10)
    months_valid_dataset = MonthDataset(data_path=data_path,
                                        df=pd.read_csv(os.path.join(data_path, "month.csv")),
                                        augment_step=0)
    print("Length Train Dataset: ", len(month_train_dataset))
    print("Length Valid Dataset: ", len(months_valid_dataset))

    IDX = random.randint(0, len(month_train_dataset)-1)
    print(IDX)
    print("Image Label: ", month_train_dataset[IDX]["label"])
    plt.figure(figsize=(4,4)); plt.title("Original Image")
    plt.imshow(month_train_dataset[IDX]["image"][0], cmap="gray")
    plt.show()

    months_data_manager = MonthDataModule(encode_dict=month_to_idx, dataset={"train_data": month_train_dataset,
                                                                             "valid_data": months_valid_dataset},
                                          test_set=None,
                                          saved_db_folder=SPLIT_FOL,
                                          batch_size=BATCH_SIZE, reload_data=False, drop_last=True)
plt.imshow(next(iter(months_data_manager.train_dataloader()))["image"].pixel_values[0][0], cmap="gray")



RESUME_TRAINING = cfg["training"]["resume" ]  
GRADIENT_ACCUMULATION = cfg["training"]["grad_accumulation"]  
CKPT_PATH = cfg["training"]["ckpt_path"]
PROJECT_NAME = cfg["training"]["wandb_proj"]
RUN_NAME = cfg["training"]["wandb_run"]
# LAST_RUN_ID is discovered once the project is created

config = {"batch_size": BATCH_SIZE,
          "learning_rate": 0.001,
          "gradient_accumulation": ("YES", GRADIENT_ACCUMULATION) if GRADIENT_ACCUMULATION > 1 else ("NO", None)}

if RESUME_TRAINING == True:
    LAST_RUN_ID = cfg["training"]["wandb_runID"]
    run = wandb.init(project=PROJECT_NAME,
                     name=RUN_NAME, config=config,
                     resume=True, id=LAST_RUN_ID)
    LAST_EPOCHS = cfg["training"]["last_epoch"]
    CKPT = os.path.join(model_ckpt_path, f"{RUN_NAME}_{LAST_RUN_ID}/model-epoch={LAST_EPOCHS - 1}.ckpt")
    ADDITTIONAL_EPOCHS = cfg["training"]["add_epoch"]

    EPOCHS = LAST_EPOCHS + ADDITTIONAL_EPOCHS

else:
    run = wandb.init(project=PROJECT_NAME,
                     name=RUN_NAME, config=config)
    
    EPOCHS = cfg["training"]["epochs"]


# Note: MPS Backend doesn't support torch.DoubleTensor(=float64)
if DEVICE.type == "mps":    model = MonthDigitalizer(in_chan=1, out_dict=idx_to_month).to(device=DEVICE, dtype=torch.float32)
else:                       model = MonthDigitalizer(in_chan=1, out_dict=idx_to_month).to(device=DEVICE)
print(model)

wandb_logger = WandbLogger(name=RUN_NAME,
                           save_dir=CKPT_PATH, offline=False,
                           project=PROJECT_NAME, log_model=False) # log_model = "all"/True/False

trainer = pl.Trainer(accelerator=DEVICE.type, # gpu, cpu, mps
                     num_sanity_val_steps=2,
                     accumulate_grad_batches=GRADIENT_ACCUMULATION,
                     logger=wandb_logger,
                     devices=1, max_epochs=EPOCHS,
                     callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.001, patience=12, mode="min"),
                                EarlyStopping(monitor="train_f1", min_delta=0.001, patience=12, mode="max"),
                                ModelCheckpoint(dirpath=f"{CKPT_PATH}/{RUN_NAME}_{wandb.run.id}",
                                                filename='model-{epoch}',
                                                save_top_k=-1)],
                     log_every_n_steps=1)

if RESUME_TRAINING == True:
    trainer.fit(model, datamodule=months_data_manager, ckpt_path=CKPT)
else:
    trainer.fit(model, datamodule=months_data_manager)
print("Training Complete")
wandb.finish()
print("End of the training program")