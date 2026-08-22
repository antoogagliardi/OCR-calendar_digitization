import collections
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import os
import yaml
from pprint import pprint
import random
import matplotlib.pyplot as plt
import wandb

from src.utils.plotting import compute_class_frequency

""" Create class weights for imbalanced class scenarios to be used in training a machine learning model.
    Parameters:
        - length_data (int): Total number of samples in the dataset.
        - class_freq (dict): A dictionary containing the absolute frequencies of each class in the dataset.
    Returns:
        - classes_weights (torch.Tensor): A 1D tensor containing the computed class weights for each class.
            These weights are calculated to address class imbalance during model training.
"""
def classes_weigths_creation(length_data:int, class_freq:dict):
    classes_weights = np.empty(shape=(0),dtype=np.float32)
    for key, freq in class_freq.items():
        epsilon = 1.0 / (length_data * np.sqrt(freq*length_data))
        weight = 1.0 / (freq + epsilon)
        classes_weights = np.append(classes_weights, [weight], axis=0)
        print(f"{key}: {np.round(freq, 7)} -> weigth= {weight}")
    classes_weights = torch.from_numpy(classes_weights).type(torch.FloatTensor)
    
    return classes_weights

def compute_class_frequency(dataset):
    total_samples = len(dataset)

    frequencies = collections.Counter()
    for entry in dataset:
        label = entry["label"]
        for lab in label:
            frequencies[lab] += 1 
    frequencies = dict(sorted(frequencies.items())) # Sort dictionary keys
  
    scaled_frequencies = {}
    for key, val in frequencies.items():
        scaled_frequencies[key] = val/total_samples

    return total_samples, frequencies, scaled_frequencies



from src.data.iamdb import CharDataset
from src.data.iamloader import IAMDataModule
from src.model.word_digitalizer import Digitalizer

print("Libraries imported")


# Read configuration file
with open("configs/config_digitText.yaml") as f:
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
plt.imshow(iam_data_manager.train_dataset[IDX]["image"][0], cmap="gray")




## Inspect Labels Frequency after Splitting
print("Training Dataset Classes Distribution")
train_samples, train_freq, scaled_train_freq = compute_class_frequency(iam_data_manager.train_dataset)
print("Length of dictionary: ", len(train_freq))
pprint(train_freq)



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
if DEVICE.type == "mps":    model = Digitalizer().to(device=DEVICE, dtype=torch.float32)
else:                       model = Digitalizer().to(device=DEVICE)
print(model)

wandb_logger = WandbLogger(name=RUN_NAME,
                           save_dir=CKPT_PATH, offline=False,
                           project=PROJECT_NAME, log_model=False) # log_model = "all"/True/False

trainer = pl.Trainer(accelerator=DEVICE.type, # gpu, cpu, mps
                     num_sanity_val_steps=2,
                     limit_train_batches=50,
                     limit_val_batches=5,
                     accumulate_grad_batches=GRADIENT_ACCUMULATION,
                     logger=wandb_logger,
                     devices=1, max_epochs=EPOCHS,
                     callbacks=[EarlyStopping(monitor="valid_loss", min_delta=0.001, patience=5, mode="min"),
                                EarlyStopping(monitor="valid_cer", min_delta=0.001, patience=5, mode="min"),
                                EarlyStopping(monitor="valid_wer", min_delta=0.001, patience=5, mode="min"),
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



