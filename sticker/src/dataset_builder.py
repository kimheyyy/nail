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
        
        image = self.apply_clahe(image)
        
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
    
    def apply_clahe(self, image):
        # CLAHE 적용 함수
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if image.ndim == 3 and image.shape[2] == 3:  # 컬러 이미지인 경우
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # LAB 컬러 스페이스로 변환
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:  # 그레이스케일 이미지인 경우
            image = clahe.apply(image)
        return image
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)
