# Libraries Import
import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os




class CalendarDataModule(pl.LightningDataModule):
    def __init__(self, dataset:dict[Dataset], test_set:Dataset, saved_db_folder:str,
                 batch_size:int=2, reload_data:bool=False, drop_last:bool=True, device:str="cpu"):
        super(CalendarDataModule, self).__init__()
        self.batch_size = batch_size
        self.drop_last_batch = drop_last

        if reload_data == True:
            if f'calendars_trainDataset.db' in os.listdir(saved_db_folder) and f'calendars_validDataset.db' in os.listdir(saved_db_folder):
                print('-- Loading existing dataset --')
                training_data = torch.load(os.path.join(saved_db_folder, f'calendars_trainDataset.db'), weights_only=False)
                validation_data = torch.load(os.path.join(saved_db_folder, f'calendars_validDataset.db'), weights_only=False)
                
                self.train_dataset = training_data["data"]; print("  - Training Dataset Length: ", len(self.train_dataset))
                self.valid_dataset = validation_data["data"]; print("  - Validation Dataset Length: ", len(self.valid_dataset))
                print("  - Data Loaded")
            else: print("Datasets not found")

            if f'calendars_testDataset.db' in os.listdir(saved_db_folder):
                self.test_dataset = torch.load(os.path.join(saved_db_folder, f"calendars_testDataset.db"), weights_only=False)
                print("Test Data Loaded")
        else:
            if dataset != None:
                # At this step we may have to perform a dataset split
                print("-- Splitting the entire dataset into Training and Validation Sets --")
                seed = torch.Generator(device="cpu").seed()
                seed = torch.Generator(device="cpu").manual_seed(seed)
                print("  - Random Torch Seed: ", seed.initial_seed())
                initial_dataset_length = len(dataset)                       ;   print("Lenght dataset: ", initial_dataset_length)
                training_length = int((2/3)*len(dataset))                   ;   print("  - Training Dataset Length: ", training_length)
                validation_length = int(len(dataset) - training_length)     ;   print("  - Validation Dataset Length: ", validation_length)
                
                self.train_dataset, self.valid_dataset = random_split(dataset,
                                                                    [training_length, validation_length], generator=seed)
            

                torch.save({"data": self.train_dataset}, os.path.join(saved_db_folder, f'calendars_trainDataset.db'))
                torch.save({"data": self.valid_dataset}, os.path.join(saved_db_folder, f'calendars_validDataset.db'))
                if self.train_dataset and self.valid_dataset: print("-- Dataset loading has been performed correctly --")
                else: print("-- Something went wrong during the dataset loading --")
            else:   print("No Dataset Valid Path has been provided")
            
            if test_set != None:
                self.test_dataset = test_set; torch.save(self.test_dataset, os.path.join(saved_db_folder, f"calendars_testDataset.db"))
                print("  - Training Dataset Length: ", len(self.test_dataset))
    
    def collate_data(self, samples):
        image = torch.FloatTensor([sample["image"] for sample in samples])
        ground_mask = torch.FloatTensor([sample["label"] for sample in samples])

        return {"image": image, "ground_mask": ground_mask}
    
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