# Libraries Import
from transformers import BertTokenizerFast
from transformers import ViTImageProcessor
from transformers.image_utils import PILImageResampling
import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os




class IAMDataModule(pl.LightningDataModule):
    def __init__(self, dataset:dict[Dataset], test_set:Dataset, iam_type:str, saved_db_folder:str,
                 batch_size:int=2, reload_data:bool=False, drop_last:bool=True):
        super(IAMDataModule, self).__init__()
        self.batch_size = batch_size
        self.drop_last_batch = drop_last
        self.iam_type = iam_type
        self.tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-cased")
        self.image_processor = ViTImageProcessor(do_resize= True,
                                                 size = [128, 512] if (iam_type == "lines" or iam_type == "sentences") else [224, 224], 
                                                 resample=PILImageResampling.BILINEAR, 
                                                 do_rescale=True, 
                                                 rescale_factor= 1 / 255, 
                                                 do_normalize= True, 
                                                 image_mean= [0.5], 
                                                 image_std= [0.5] )


        if reload_data == True:
            if f'iam_{self.iam_type}_trainDataset.db' in os.listdir(saved_db_folder) and f'iam_{self.iam_type}_validDataset.db' in os.listdir(saved_db_folder):
                print('-- Loading existing dataset --')
                training_data = torch.load(os.path.join(saved_db_folder, f'iam_{self.iam_type}_trainDataset.db'), weights_only=False)
                validation_data = torch.load(os.path.join(saved_db_folder, f'iam_{self.iam_type}_validDataset.db'), weights_only=False)
                
                self.train_dataset = training_data["data"]; print("  - Training Dataset Length: ", len(self.train_dataset))
                self.valid_dataset = validation_data["data"]; print("  - Validation Dataset Length: ", len(self.valid_dataset))
                print("  - Data Loaded")
            else: print("Datasets not found")

            if f'iam_{self.iam_type}_testDataset.db' in os.listdir(saved_db_folder):
                self.test_dataset = torch.load(os.path.join(saved_db_folder, f"iam_{self.iam_type}_testDataset.db"), weights_only=False)
                print("Test Data Loaded")
        else:
            if dataset != None:
                # At this step we may have to perform a dataset split
                print("-- Splitting the entire dataset into Training and Validation Sets --")
                seed = torch.Generator(device="cpu").seed()
                seed = torch.Generator(device="cpu").manual_seed(seed)
                print("  - Random Torch Seed: ", seed.initial_seed())
                initial_dataset_length = len(dataset)
                print("Lenght dataset: ", initial_dataset_length)
                training_length = int((2/3)*len(dataset))
                print("  - Training Dataset Length: ", training_length)
                validation_length = int(len(dataset) - training_length)
                print("  - Validation Dataset Length: ", validation_length)
                
                self.train_dataset, self.valid_dataset = random_split(dataset,
                                                                    [training_length, validation_length], generator=seed)
            

                torch.save({"data": self.train_dataset}, os.path.join(saved_db_folder, f'iam_{self.iam_type}_trainDataset.db'))
                torch.save({"data": self.valid_dataset}, os.path.join(saved_db_folder, f'iam_{self.iam_type}_validDataset.db'))
                if self.train_dataset and self.valid_dataset: print("-- Dataset loading has been performed correctly --")
                else: print("-- Something went wrong during the dataset loading --")
            else: print("No Dataset Valid Path has been provided")
            
            if test_set != None:
                self.test_dataset = test_set; torch.save(self.test_dataset, os.path.join(saved_db_folder, f"iam_{self.iam_type}_testDataset.db"))
                print("  - Training Dataset Length: ", len(self.test_dataset))
    
    def collate_data(self, samples):
        if self.iam_type == "sentences" or self.iam_type == "lines":
            image = self.image_processor([sample["image"] for sample in samples],
                                         return_tensors="pt")
            ground_mask = torch.FloatTensor([sample["mask"] for sample in samples])

            return {"image": image, "ground_mask": ground_mask}
        if self.iam_type == "words":
            image = self.image_processor([sample["image"] for sample in samples],
                                         return_tensors="pt")
            batch_max_length = max([len(sample["label"]) for sample in samples])
            label = self.tokenizer([" ".join(sample["label"]) for sample in samples],
                                   padding=True, max_length=batch_max_length,
                                   truncation=False,
                                   add_special_tokens=True,
                                   return_tensors="pt")
            
            # IMPORTANT: make sure that PAD tokens are ignored by the loss function
            for row in label.input_ids:
                for i in range(len(row)):
                    if row[i] != self.tokenizer.pad_token_id: continue
                    else: row[i] = -100
            
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