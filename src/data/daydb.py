import lightning.pytorch as pl
import os
import numpy as np
import cv2 as cv2
import pandas as pd
from transformers import ViTImageProcessor
from transformers.image_utils import PILImageResampling
from imgaug import augmenters as iaa
from tqdm import tqdm


class NumberDataset(pl.LightningDataModule):
    def __init__(self, data_path:str, df:pd.DataFrame, augment_step:int=0):
        super(NumberDataset, self).__init__()
        self.image_processor = ViTImageProcessor(do_resize= True,
                                                 size = [32, 32], 
                                                 resample=PILImageResampling.BILINEAR, 
                                                 do_rescale=True, 
                                                 rescale_factor= 1 / 255, 
                                                 do_normalize= True, 
                                                 image_mean= [0.5], 
                                                 image_std= [0.5] )
        self.seq = iaa.Sequential([
            # iaa.Sometimes(0.3, iaa.Affine(scale=(0.5, 1.5))),
            iaa.Sometimes(0.3, iaa.Affine(translate_percent=(-0.20, 0.20))),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),
            iaa.Sometimes(0.5, iaa.Dropout(p=(0, 0.30))),
            iaa.Sometimes(0.45, iaa.GaussianBlur(sigma=(0.0, 1.0))),
            # iaa.Affine(shear=(-5,5)),
            # iaa.Affine(rotate=(-45,45)),

        ], random_order=False)
        
        self.data = []
        with tqdm(range(len(df)), desc="Dataset Creation") as pbar:
            for row in range(len(df)):
                if os.path.getsize(os.path.join(data_path, df["img"][row])) != 0:
                    self.data.append([os.path.join(data_path, df["img"][row]),
                                    str(df["label"][row])])
                    pbar.update()
        
        self.create_dataset()
        self.augment_data(aug_step=augment_step)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        return {"image": self.data[i][0],
                "label": self.data[i][1]}
    
    def create_dataset(self):
        for i in range(len(self.data)):
            path = self.data[i][0]
            image = self.read_image(img_path=path)
            self.data[i][0] = image
    
    def augment_data(self, aug_step:int=1):
        augmentation = []
        with tqdm(range(aug_step), desc="Augmentation Process") as pbar:
            for _, i in zip(pbar, range(aug_step)):
                augmented_train = [[np.transpose(self.seq(image=np.transpose(self.data[i][0], axes=[1,2,0])),
                                                          axes=[2,0,1]),
                                    self.data[i][1]]
                                    for i in range(len(self.data))]
                if i == 0: augmentation = augmented_train
                else: augmentation.extend(augmented_train)
                pbar.set_postfix({"AUGMENTATION STEP": i+1})
                pbar.update()

        self.data.extend(augmentation)  
   
    def read_image(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype("uint8")
        image = np.expand_dims(image, axis=0)
        
        return image