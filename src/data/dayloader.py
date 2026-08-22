import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor
from transformers.image_utils import PILImageResampling
import os




class NumberDataModule(pl.LightningDataModule):
    def __init__(self, encode_dict:dict, dataset:dict[Dataset], test_set:Dataset, saved_db_folder:str,
                 batch_size:int=2, reload_data:bool=False, drop_last:bool=True):
        super(NumberDataModule, self).__init__()
        self.encode_dict = encode_dict
        self.batch_size = batch_size
        self.drop_last_batch = drop_last
        self.image_processor = ViTImageProcessor(do_resize= True,
                                                 size = [32, 32], 
                                                 resample=PILImageResampling.BILINEAR, 
                                                 do_rescale=True, 
                                                 rescale_factor= 1 / 255, 
                                                 do_normalize= True, 
                                                 image_mean= [0.5], 
                                                 image_std= [0.5] )


        if reload_data == True:
            if f'numbers_trainDataset.db' in os.listdir(saved_db_folder) and f'numbers_validDataset.db' in os.listdir(saved_db_folder):
                print('-- Loading existing dataset --')
                training_data = torch.load(os.path.join(saved_db_folder, f'numbers_trainDataset.db'), weights_only=False)
                validation_data = torch.load(os.path.join(saved_db_folder, f'numbers_validDataset.db'), weights_only=False)
                
                self.train_dataset = training_data["data"]  ; print("  - Training Dataset Length: ", len(self.train_dataset))
                self.valid_dataset = validation_data["data"]; print("  - Validation Dataset Length: ", len(self.valid_dataset))
                
                print("  - Data Loaded")
            else: print("Datasets not found")

            if f'numbers_testDataset.db' in os.listdir(saved_db_folder):
                self.test_dataset = torch.load(os.path.join(saved_db_folder, f"numbers_testDataset.db"), weights_only=False)
                print("Test Data Loaded")
        else:
            if dataset != None:
                self.train_dataset = dataset["train_data"]
                self.valid_dataset = dataset["valid_data"]
                
                torch.save({"data": self.train_dataset}, os.path.join(saved_db_folder, f'numbers_trainDataset.db'))
                torch.save({"data": self.valid_dataset}, os.path.join(saved_db_folder, f'numbers_validDataset.db'))
                if self.train_dataset and self.valid_dataset: print("-- Dataset loading has been performed correctly --")
                else: print("-- Something went wrong during the dataset loading --")
            else: print("No Dataset Valid Path has been provided")
            
            if test_set != None:
                self.test_dataset = test_set; torch.save(self.test_dataset, os.path.join(saved_db_folder, f"numbers_testDataset.db"))
                print("  - Training Dataset Length: ", len(self.test_dataset))
    
    def collate_data(self, samples):
        image = self.image_processor([sample["image"] for sample in samples],
                                         return_tensors="pt")
        label = torch.LongTensor([self.encode_dict[sample["label"]] for sample in samples])

        return {"image": image, "label": label}
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.collate_data,
                                      drop_last=True if self.drop_last_batch == True else False)
        return train_dataloader
    
    def val_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                      shuffle=False, collate_fn=self.collate_data,
                                      drop_last=True if self.drop_last_batch == True else False)
        return valid_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False, collate_fn=self.collate_data,
                                      drop_last=True if self.drop_last_batch == True else False)
        return test_dataloader   