import os
import shutil
import random
import math
from glob import glob
from ultralytics import YOLO
import zipfile
import albumentations as A
import cv2
import numpy as np

# === MedAugment 核心函數（嚴格遵循原論文 Document 3 參數）===
def odd_conversion(num):
    """確保回傳奇數（用於 kernel size）"""
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num

def get_medaugment_transform(level=5):
    """
    原始 MedAugment 論文實作（Document 3）
    完整保留所有參數設定，適配 YOLO 物件偵測
    
    關鍵特性：
    - 所有 Affine 變換使用 keep_ratio=True
    - Shear/Translate 為單向 (0, X)
    - 完整的 Affine 參數（interpolation, mode, cval 等）
    
    Args:
        level: 1-10，控制增強強度（預設5為中等強度）
    """
    return A.Compose([
        # === Pixel-level 變換 (前6個) ===
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, odd_conversion(3 + 0.8 * level)), p=0.2 * level),
        
        #  保持原論文的 var_limit（雖然不是標準參數，但這是原實作）
        A.GaussNoise(std_range=(0.002 * level, 0.01 * level), per_channel=True, p=0.2 * level),
        
        # === Spatial-level 變換 (後8個) ===
        A.Rotate(
            limit=4 * level, 
            interpolation=1, 
            border_mode=0, 
            value=0, 
            mask_value=None, 
            rotate_method='largest_box',
            crop_border=False, 
            p=0.2 * level
        ),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        
        # Scale - 只設定 scale 參數，省略其他
        A.Affine(
            scale=(1 - 0.04 * level, 1 + 0.04 * level),
            interpolation=1,
            mask_interpolation=0,
            cval=0,
            cval_mask=0,
            mode=0,
            fit_output=False,
            keep_ratio=True,  #  原論文設定
            p=0.2 * level
        ),
        
        # Shear X - 單向 (0, 2*level)
        A.Affine(
            shear={'x': (0, 2 * level), 'y': (0, 0)},
            interpolation=1,
            mask_interpolation=0,
            cval=0,
            cval_mask=0,
            mode=0,
            fit_output=False,
            keep_ratio=True,  #  原論文設定
            p=0.2 * level
        ),
        
        # Shear Y - 單向 (0, 2*level)
        A.Affine(
            shear={'x': (0, 0), 'y': (0, 2 * level)},
            interpolation=1,
            mask_interpolation=0,
            cval=0,
            cval_mask=0,
            mode=0,
            fit_output=False,
            keep_ratio=True,  #  原論文設定
            p=0.2 * level
        ),
        
        # Translate X - 單向 (0, 0.02*level)
        A.Affine(
            translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)},
            interpolation=1,
            mask_interpolation=0,
            cval=0,
            cval_mask=0,
            mode=0,
            fit_output=False,
            keep_ratio=True,  #  原論文設定
            p=0.2 * level
        ),
        
        # Translate Y - 單向 (0, 0.02*level)
        A.Affine(
            translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)},
            interpolation=1,
            mask_interpolation=0,
            cval=0,
            cval_mask=0,
            mode=0,
            fit_output=False,
            keep_ratio=True,  #  原論文設定
            p=0.2 * level
        ),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def apply_medaugment_branch(img, bboxes, class_labels, level=5, number_branch=4):
    """
    應用 MedAugment 分支策略（原論文實作）
    
    Args:
        img: 輸入影像 (numpy array)
        bboxes: YOLO 格式的 bboxes [[x_center, y_center, w, h], ...]
        class_labels: 類別標籤 [0, 0, 1, ...]
        level: 增強強度 (1-10)
        number_branch: 生成的分支數量
    
    Returns:
        list of (augmented_image, augmented_bboxes, augmented_labels)
    """
    transform = get_medaugment_transform(level)
    
    # MedAugment 分支策略：(pixel-level數量, spatial-level數量)
    strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
    
    results = []
    strategy_copy = strategy.copy()
    
    for i in range(number_branch):
        # 選擇策略（原論文邏輯）
        if number_branch != 4:
            employ = random.choice(strategy)
        else:
            if len(strategy_copy) == 0:
                strategy_copy = strategy.copy()
            index = random.randrange(len(strategy_copy))
            employ = strategy_copy.pop(index)
        
        # 從變換中採樣
        pixel_transforms = random.sample(transform.transforms[:6], employ[0])
        spatial_transforms = random.sample(transform.transforms[6:], employ[1])
        
        # 組合並隨機打亂（原論文邏輯）
        selected_transforms = pixel_transforms + spatial_transforms
        branch_transform = A.Compose(
            selected_transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        random.shuffle(branch_transform.transforms)
        
        try:
            transformed = branch_transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            results.append((
                transformed['image'],
                transformed['bboxes'],
                transformed['class_labels']
            ))
        except Exception as e:
            print(f"       分支 {i} 增強失敗: {e}")
            continue
    
    return results

def augment_image_medaugment(img_path, lbl_path, output_img_dir, output_lbl_dir, 
                              level=5, number_branch=4, save_original=True):
    """
    對單一影像應用 MedAugment
    """
    # 讀取影像
    img = cv2.imread(img_path)
    if img is None:
        print(f"       無法讀取 {img_path}")
        return
    
    # 讀取 YOLO 標籤
    with open(lbl_path, 'r') as f:
        labels = f.readlines()
    
    bboxes = []
    class_labels = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                class_id = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:5])
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
            except (ValueError, IndexError) as e:
                print(f"       標籤格式錯誤: {line.strip()} - {e}")
                continue
    
    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
    ext = os.path.basename(img_path).rsplit('.', 1)[1]
    
    # 應用 MedAugment 分支策略
    augmented_results = apply_medaugment_branch(img, bboxes, class_labels, level, number_branch)
    
    # 儲存增強結果
    for idx, (aug_img, aug_bboxes, aug_labels) in enumerate(augmented_results):
        # 儲存影像
        aug_img_name = f"{base_name}_{idx+1}.{ext}"
        aug_img_path = os.path.join(output_img_dir, aug_img_name)
        cv2.imwrite(aug_img_path, aug_img)
        
        # 儲存標籤
        aug_lbl_name = f"{base_name}_{idx+1}.txt"
        aug_lbl_path = os.path.join(output_lbl_dir, aug_lbl_name)
        with open(aug_lbl_path, 'w') as f:
            for cls, bbox in zip(aug_labels, aug_bboxes):
                f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    
    # 儲存原始影像（作為第 number_branch+1 個分支）
    if save_original and len(augmented_results) > 0:
        orig_img_name = f"{base_name}_{number_branch+1}.{ext}"
        orig_lbl_name = f"{base_name}_{number_branch+1}.txt"
        shutil.copy(img_path, os.path.join(output_img_dir, orig_img_name))
        shutil.copy(lbl_path, os.path.join(output_lbl_dir, orig_lbl_name))