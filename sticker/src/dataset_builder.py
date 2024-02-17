#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""
import torch
import os
import cv2
from data_visualizer import one_hot_encode


class NailDataset(torch.utils.data.Dataset):

    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        import cv2
        import os

        # 이미지 파일 경로 검증 및 불러오기
        image_path = self.image_paths[i]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식일 수 있습니다: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 마스크 파일 경로 검증 및 불러오기
        mask_path = self.mask_paths[i]
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"마스크 파일을 찾을 수 없습니다: {mask_path}")
        mask = cv2.imread(mask_path)
        if mask is None:
            raise ValueError(f"마스크를 불러올 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식일 수 있습니다: {mask_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)
