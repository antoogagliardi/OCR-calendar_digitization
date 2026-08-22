# Libraries Imported
from src.utils.utility import resize_image, retrieve_relevant_px_value

import numpy as np
import lightning.pytorch as pl
from tqdm import tqdm
import os
import re
import cv2 as cv2
import pandas as pd




class CharDataset(pl.LightningDataModule):
    def __init__(self, data_path, iam_type:str="words", mode:str="BW"):
        super(CharDataset, self).__init__()
        self.mode = mode
        if iam_type == "lines":     self.iam = os.path.join(data_path, "lines.csv")     ; self.iam_type = "lines"
        if iam_type == "sentences": self.iam = os.path.join(data_path, "sentences.csv") ; self.iam_type = "sentences"
        if iam_type == "words":     self.iam = os.path.join(data_path, "words.csv")     ; self.iam_type = "words"

        if (iam_type == "lines" or iam_type == "sentences"):
            self.height = 128   ;   self.width = 512
        if iam_type == "words":
            self.height = 224   ;   self.width = 224
        
        dataframe = pd.read_csv(os.path.join(data_path, self.iam))
        CHARS_TO_REMOVE = '!"#%&\'()*+,-./:;?'                          # Character to be removed from original string
        TRANSLATIONAL_TABLE = str.maketrans('', '', CHARS_TO_REMOVE)
        self.data = []                                                  # List of data (Image, Label, Gray_Level)
        with tqdm(range(len(dataframe)), desc="Data Retrieval") as pbar:
            for row in range(len(dataframe)):
                if str(dataframe["ground_truth"][row]).translate(TRANSLATIONAL_TABLE) != "":
                    if os.path.getsize(os.path.join(data_path, dataframe["img"][row])) != 0:
                        self.data.append([os.path.join(data_path, dataframe["img"][row]),
                                          str(dataframe["ground_truth"][row]).lower().translate(TRANSLATIONAL_TABLE),
                                          dataframe["gray_level"][row]])
                        pbar.update()

        # Dataset Creation
        self.create_dataset()

        # Ground Truth length
        lenghts = [len(row[1]) for row in self.data]
        print("minimum label length: ", min(lenghts))
        print("maximum label length: ", max(lenghts))
        print("median label length: ", int(sum(lenghts)/len(lenghts)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        if self.iam_type == "sentences" or self.iam_type == "lines":
            return {"image": self.data[i][0],
                    "label": self.data[i][1],
                    "mask": self.data[i][3]}
        if self.iam_type == "words":
            return {"image": self.data[i][0],
                    "label": self.data[i][1]}
    
    def create_dataset(self):
        with tqdm(range(len(self.data)), desc="Dataset Creation") as pbar:
            for i, img in zip(pbar, self.data):
                img_path = img[0]
                if self.iam_type == "lines" or self.iam_type == "sentences":
                    label = list(re.split("\s+", img[1]))
                    label = [lab for lab in label if lab != ""]
                    image, mask = self.process_iam_sentence_image(image_path=img_path, gray_level=img[2])
                    
                    self.data[i][0] = image
                    self.data[i][1] = label
                    self.data[i].append(mask)
                
                if self.iam_type == "words":
                    label = img[1]
                    chars = [*label]
                    image = self.process_iam_word_image(image_path=img_path, gray_level=img[2])
                    
                    self.data[i][0] = image
                    self.data[i][1] = chars
                
                pbar.set_postfix({"IMG": img_path, "GRAY_LV": img[2]})
                pbar.update()
    
    def process_iam_word_image(self, image_path:str, gray_level:int):
        # Open the image
        if self.mode == "RGB": image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.mode == "BW": image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype("uint8") 
        
        THRESHOLD_LIMIT = gray_level
        _, image = cv2.threshold(image, THRESHOLD_LIMIT, 255, cv2.THRESH_BINARY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = np.expand_dims(image, axis=0)

        return image
    
    def process_iam_sentence_image(self, image_path:str, gray_level:int):
        # Open the image
        if self.mode == "RGB": image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.mode == "BW": image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype("uint8")       
        # plt.figure(figsize=(25, 35))
        # plt.subplot(1, 2, 1)
        # plt.title('Original Image')
        # plt.imshow(resize_image(image=image, size=(self.height, self.width)), cmap="gray")
        # plt.axis('on')
        # plt.show
        
        ######## MASK CREATION
        detail = cv2.Laplacian(image, cv2.CV_64F, ksize=1)
        detail = resize_image(image=detail, size=(self.height, self.width))
        # plt.figure(figsize=(25, 35))
        # plt.subplot(1, 2, 1)
        # plt.title('Details')
        # plt.imshow(detail, cmap="gray")
        # plt.axis('on')
        # plt.show
        relevant_px_values = retrieve_relevant_px_value(detail)
        detail = np.where((detail > 0) & (detail <= relevant_px_values[-1]), 255, detail)
        detail = np.where((detail >= relevant_px_values[0]) & (detail < 0), 255//2, detail)
        detail = np.where((detail == 0), 0, detail)
        relevant_px_values = retrieve_relevant_px_value(detail) 
        # plt.figure(figsize=(25, 35))
        # plt.subplot(1, 2, 1)
        # plt.title('Details Corrected')
        # plt.imshow(detail, cmap="gray")
        # plt.axis('on')
        # plt.show
        mask = cv2.GaussianBlur(detail, (5, 5), 0)
        mask = np.where(mask == 0, 0, mask)
        mask = np.where(mask > 0, 255, mask)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
        mask = np.where(mask == 0, 0, mask)
        mask = np.where(mask == 255, 1, mask)
        mask = np.expand_dims(mask, axis=0)
        # plt.figure(figsize=(25, 35))
        # plt.subplot(1, 2, 1)
        # plt.title('Image Binary Mask')
        # plt.imshow(mask[0], cmap="gray")
        # plt.axis('on')
        # plt.show


        ######## IMAGE CREATION: Threshold the image (ensure uniform black and white colors)
        THRESHOLD_LIMIT = gray_level
        _, image = cv2.threshold(image, THRESHOLD_LIMIT, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # image = resize_image(image=image, size=(self.height, self.width))
        # thresh = threshold_otsu(image)
        # image = image > thresh 
        # image = np.where(image == False, 0, image)
        # image = np.where(image == True, 1, image)
        image = np.expand_dims(image, axis=0)

        
        return image, mask