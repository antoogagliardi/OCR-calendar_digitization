import yaml
import os
import numpy as np
import pandas as pd
import random
import collections
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb
from pprint import pprint

from src.data.daydb import NumberDataset
from src.data.dayloader import NumberDataModule
from src.model.day_digitizer import DayDigitalizer


from src.utils.plotting import plot_class_frequency

def classes_weigths_creation(length_data:int, class_freq:dict):
    classes_weights = np.empty(shape=(0),dtype=np.float32)
    for key, freq in class_freq.items():
        epsilon = 1.0 / (length_data * np.sqrt(freq*length_data))
        weight = 1.0 / (freq + epsilon)
        classes_weights = np.append(classes_weights, [weight], axis=0)
        print(f"{key}: {np.round(freq, 7)} -> weigth= {weight}")
    classes_weights = torch.from_numpy(classes_weights).type(torch.FloatTensor)
    
    return classes_weights

def compute_class_frequency1(dataset):
    total_samples = len(dataset)

    frequencies = collections.Counter()
    for entry in dataset:
        label = entry["label"]
        frequencies[label] += 1 
    frequencies = dict(sorted(frequencies.items())) # Sort dictionary keys
  
    scaled_frequencies = {}
    for key, val in frequencies.items():
        scaled_frequencies[key] = val/total_samples

    return total_samples, frequencies, scaled_frequencies


print("Libraries imported")


# Read configuration file
with open("configs/config_digitDay.yaml") as f:
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


# Create Days Vocabulary
MAX_NUM_DAYS = 31
RELOAD_DICT = cfg["training"]["reload_dict"]

if RELOAD_DICT == True:
    idx_to_digit = torch.load(os.path.join(model_data_path, f'digit_dictionary.dict'))["idx_digit"]
    digit_to_idx = torch.load(os.path.join(model_data_path, f'digit_dictionary.dict'))["digit_idx"]
    print("length of model vocabulary: ", len(idx_to_digit))
    print(idx_to_digit)
else:
    idx_to_digit = {}   ;   digit_to_idx = {}
    days = np.arange(MAX_NUM_DAYS)
    for day in days:
        digit_to_idx[str(day + 1)] = len(digit_to_idx)
        idx_to_digit[len(idx_to_digit)] = str(day + 1)
    pprint(digit_to_idx.keys())

    print("length of model vocabulary: ", len(idx_to_digit))
    pprint(digit_to_idx)
    
    torch.save({"idx_digit": idx_to_digit,
                "digit_idx": digit_to_idx}, os.path.join(model_data_path, f'digit_dictionary.dict'))


# Dataset Creation and Data Batching
SPLIT_FOL   = cfg["paths"]["data_split"]
RELOAD_DATA = cfg["training"]["reload_data"]
BATCH_SIZE = cfg["training"]["batch_size"]

if RELOAD_DATA == True:
    numbers_data_manager = NumberDataModule(encode_dict=digit_to_idx, dataset=None, test_set=None,
                                            saved_db_folder=SPLIT_FOL,
                                            batch_size=BATCH_SIZE, reload_data=True, drop_last=True)
else:
    number_train_dataset = NumberDataset(data_path=data_path,
                                         df=pd.read_csv(os.path.join(data_path, "num.csv")),
                                         augment_step=5)
    number_valid_dataset = NumberDataset(data_path=data_path,
                                         df=pd.read_csv(os.path.join(data_path, "num.csv")),
                                         augment_step=0)
    print("Length Train Dataset: ", len(number_train_dataset))
    print("Length Valid Dataset: ", len(number_valid_dataset))

    IDX = random.randint(0, len(number_train_dataset)-1)
    print(IDX)
    print("Image Label: ", number_train_dataset[IDX]["label"])
    plt.figure(figsize=(4,4)); plt.title("Original Image")
    plt.imshow(number_train_dataset[IDX]["image"][0], cmap="gray")
    plt.show()

    
    numbers_data_manager = NumberDataModule(encode_dict=digit_to_idx, dataset={"train_data": number_train_dataset,
                                                                               "valid_data": number_valid_dataset},
                                            test_set=None,
                                            saved_db_folder=SPLIT_FOL,
                                            batch_size=BATCH_SIZE, reload_data=False, drop_last=True)

plt.imshow(next(iter(numbers_data_manager.train_dataloader()))["image"].pixel_values[0][0], cmap="gray")



## Inspect Labels Frequency after Splitting
print("Training Dataset Classes Distribution")
train_samples, train_freq, scaled_train_freq = compute_class_frequency1(numbers_data_manager.train_dataset)

plot_class_frequency(train_freq)
training_class_weigths = classes_weigths_creation(length_data=train_samples,
                                                   class_freq=scaled_train_freq)
print("Len class weights: ", len(training_class_weigths))
pprint(training_class_weigths)




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
if DEVICE.type == "mps":    model = DayDigitalizer(in_chan=1, out_dict=idx_to_digit).to(device=DEVICE, dtype=torch.float32)
else:                       model = DayDigitalizer(in_chan=1, out_dict=idx_to_digit).to(device=DEVICE)
print(model)

wandb_logger = WandbLogger(name=RUN_NAME,
                           save_dir=CKPT_PATH, offline=False,
                           project=PROJECT_NAME, log_model=False) # log_model = "all"/True/False

trainer = pl.Trainer(accelerator=DEVICE.type, # gpu, cpu, mps
                     num_sanity_val_steps=2,
                     accumulate_grad_batches=GRADIENT_ACCUMULATION,
                     logger=wandb_logger,
                     devices=1, max_epochs=EPOCHS,
                     callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.001, patience=9, mode="min"),
                                EarlyStopping(monitor="train_f1", min_delta=0.001, patience=9, mode="max"),
                                ModelCheckpoint(dirpath=f"{CKPT_PATH}/{RUN_NAME}_{wandb.run.id}",
                                                filename='model-{epoch}',
                                                save_top_k=-1)],
                     log_every_n_steps=1)

if RESUME_TRAINING == True:
    trainer.fit(model, datamodule=numbers_data_manager, ckpt_path=CKPT)
else:
    trainer.fit(model, datamodule=numbers_data_manager)
print("Training Complete")

wandb.finish()

