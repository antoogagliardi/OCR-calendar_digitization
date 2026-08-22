import os
import random
import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
from pprint import pprint


from src.data.iamdb import CharDataset
from src.data.iamloader import IAMDataModule
from src.model.word_parser import WordParser

print("Libraries imported")


# Read configuration file
with open("configs/config_parserWord.yaml") as f:
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


# Dataset Creation and Data Batching
# Dataset Creation : IAM Handwriting Data Loading
IAM_TYPE = cfg["paths"]["data_type"]
print("IAM Handwriting dataset type: ", IAM_TYPE)

SPLIT_FOL   = cfg["paths"]["data_split"]
RELOAD_DATA = cfg["training"]["reload_data"]
BATCH_SIZE  = cfg["training"]["batch_size"]

if RELOAD_DATA == True:
    iam_data_manager = IAMDataModule(iam_type=IAM_TYPE,
                                     dataset=None, test_set=None,
                                     saved_db_folder=SPLIT_FOL, batch_size=BATCH_SIZE,
                                     reload_data=True, drop_last=True)
else:
    iam_dataset = CharDataset(data_path=data_path, iam_type=IAM_TYPE, mode="BW")
    print("dataset length: ", len(iam_dataset))

    iam_data_manager = IAMDataModule(iam_type=IAM_TYPE,
                                     dataset=iam_dataset, test_set=None,
                                     saved_db_folder=SPLIT_FOL, batch_size=BATCH_SIZE,
                                     reload_data=False, drop_last=True)
    

iam_batch_prova = next(iter(iam_data_manager.train_dataloader()))
pprint(iam_batch_prova)

IDX = random.randint(0, len(iam_data_manager.train_dataset)-1)
print("Ground Label: ", iam_data_manager.train_dataset[IDX]["label"])
plt.imshow(iam_data_manager.train_dataset[IDX]["image"][0],
           cmap="gray")



# Model Training
RESUME_TRAINING = cfg["training"]["resume" ]
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
if DEVICE.type == "mps":    model = WordParser(n_channels=1, n_classes=3, bilinear=False).to(device=DEVICE, dtype=torch.float32)
else:                       model = WordParser(n_channels=1, n_classes=3, bilinear=False).to(device=DEVICE)
print(model)

wandb_logger = WandbLogger(name=RUN_NAME,
                           save_dir=CKPT_PATH, offline=False,
                           project=PROJECT_NAME, log_model=False) # log_model = "all"/True/False

trainer = pl.Trainer(accelerator=DEVICE.type, # gpu, cpu, mps
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
    trainer.fit(model, datamodule=iam_data_manager, ckpt_path=CKPT)
else:
    trainer.fit(model, datamodule=iam_data_manager)
print("Training Complete")
wandb.finish()
print("End of the training program")