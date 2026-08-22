# Libraries Import
from src.utils.utility import align_images, read_vgg_annotation

import lightning.pytorch as pl
import numpy as np
import cv2 as cv2
import os
import re
import json
from tqdm import tqdm

    

    
class CalendarDataset(pl.LightningDataModule):
    def __init__(self, data_path, mode:str="BW", downscale:bool=False):
        super(CalendarDataset, self).__init__()
        self.label_to_color = {"none": 0,
                               "month": 1,
                               "day": 2,
                               "note": 3,
                               "emptyNote": 4,
                               "border": 5}
        self.calendar_template = cv2.imread(os.path.join(data_path, "calendars_template_original_size.png"))
        self.mode = mode
        self.downscale = downscale
        self.pyr_lev = 2
        
        self.data = []
        calendars_path = os.path.join(data_path, "calendars")
        ann_path = os.path.join(data_path, "calendars_masks")
        
        imgs = sorted(os.listdir(calendars_path))
        print("Calendar imgs: ", imgs)
        if ".DS_Store" in imgs: imgs.remove(".DS_Store")
        with tqdm(range(len(imgs)), desc="Data retrieval") as pbar:
            for _, img in zip(pbar, imgs):
                month = re.split("_", img)
                suffix = re.split("[_.]", img)[-2]
                ground_ann = "_".join(month[0:1]) + f"_{suffix}" + ".json"

                self.data.append([img,
                                  month[0],
                                  os.path.join(calendars_path, img),
                                  os.path.join(ann_path, ground_ann)])
                pbar.set_postfix({"IMG": img, "GROUND": ground_ann, "SUFFIX":suffix})
                pbar.update()

        self.create_dataset()
        print("Calendar Dataset Length: ", len(self.data))
        # self.augment_dataset()
        # print("Calendar Dataset Length after Augmentation: ", len(self.data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        id = self.data[i][0]
        month = self.data[i][1]
        image_i = self.data[i][2]
        ground_i = self.data[i][3]
        return {"id": id,
                "month": month,
                "image": image_i,
                "label": ground_i }
    
    def seg_mask_creation(self, annotations):
        # Define the image dimensions (replace with actual dimensions)
        image_height = self.calendar_template.shape[0] // (2 * self.pyr_lev if self.downscale == True else 1)
        image_width = self.calendar_template.shape[1] // (2 * self.pyr_lev  if self.downscale == True else 1)

        # Create a black image
        ground_truth = np.zeros((image_height, image_width), dtype=np.uint8)
        for ann in annotations:
            width = ann.width // (2 * self.pyr_lev if self.downscale == True else 1);   height = ann.height // (2 * self.pyr_lev if self.downscale == True else 1)
            x1 = ann.x // (2 * self.pyr_lev if self.downscale == True else 1);          y1 = ann.y // (2 * self.pyr_lev if self.downscale == True else 1)
            
            label = ann.type
        
            x2 = abs(width + x1); y2 = abs(height + y1)

            points = [x1, x2, y1, y2]     
            
            # Draw on the ground truth image
            ground_truth[y1:y2, x1:x2] = self.label_to_color[label]
        
        ground_truth = np.expand_dims(ground_truth, axis=0)
        return ground_truth

    def create_dataset(self):
        images = []
        with tqdm(range(len(self.data)), desc="Dataset Creation") as pbar:
            for _, img in zip(pbar, self.data):
                image = self.open_img(img[2])
                with open(img[3], mode="r") as file:
                    annotations = json.load(file)
                annotations = read_vgg_annotation(json_data=annotations)
                ground_truth = self.seg_mask_creation(annotations)
                
                images.append([img[0], img[1], image, ground_truth])
                pbar.update()
        self.data = images

    def open_img(self, img_path):
        if self.mode == "RGB":
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = align_images(image=image, template=self.calendar_template, MAX_FEATURES=1000, KEEP_PERCENT=0.2, debug=False)
            if self.downscale == True:
                for i in range(self.pyr_lev):
                    image = cv2.pyrDown(image, dstsize=(image.shape[1] // 2, image.shape[0] // 2))
                    ground = cv2.pyrDown(ground, dstsize=(ground.shape[1] // 2, ground.shape[0] // 2))
            image = np.expand_dims(image, axis=0)
        
        if self.mode == "BW":
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = align_images(image=image, template=self.calendar_template, MAX_FEATURES=1000, KEEP_PERCENT=0.2, debug=False)
            if self.downscale == True:
                for i in range(self.pyr_lev):
                    image = cv2.pyrDown(image, dstsize=(image.shape[1] // 2, image.shape[0] // 2))
            image = np.expand_dims(image, axis=0)

        return image
    
    # def augment_dataset(self):
    #     images_aug = []
    #     for idx in range(len(self.data)):
    #         data_i = self.data[idx]
    #         images_aug.extend([[data_i[0],
    #                             data_i[1],
    #                             self.seq(image=data_i[2]),
    #                             data_i[3],
    #                             ]])
    #     print("LENGTH AUGMENTED LIST: ", len(images_aug))
    #     self.data.extend(images_aug)