"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

#conda activate mmdet
from PIL import Image
#import imgviz
#pip install imgviz
import cv2
#conda install opencv
import argparse
import os
import numpy as np
#import tqdm
#conda install tqdm

import tensorflow.compat.v1 as tf
from third_party.deeplab.core import feature_extractor
from core import preprocess_utils


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img

def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src[:,:,0], dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)
    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    #size of output in cv2.resize() is defined as (w,h)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)

    if len(sub_img01.shape) == 2:
        sub_img01 = np.expand_dims(sub_img01, axis=2)
    if len(sub_img02.shape) == 2:
        sub_img02 = np.expand_dims(sub_img02, axis=2)

    sub_img01 = cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
               interpolation=cv2.INTER_NEAREST)
    if len(sub_img01.shape) == 2:
        sub_img01 = np.expand_dims(sub_img01, axis=2)

    img_main = img_main - sub_img02 + sub_img01
    #img_main = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
    #                                             interpolation=cv2.INTER_NEAREST)
    return img_main

def rescale_src(mask_src, img_src, h, w):
    #rescale and paste randomly
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    mask_src = np.expand_dims(mask_src, axis=2)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w, 1), dtype=np.uint8)
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


def Large_Scale_Jittering(mask, img, min_scale=0.1, max_scale=2.0):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    #h = int(h)
    #w = int(w)
    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)

    #img, mask = preprocess_utils.randomly_scale_image_and_label(img, mask, rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, axis=2)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)
    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))

    ##mask: Label with shape [height, width, 1].
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        #RGB(168,168,168) is grey
        mask_pad = np.zeros((h, w, 1), dtype=np.uint8)
        #
        #img_pad = np.ones((h, w, 3), dtype=np.float32) * 168
        #mask_pad = np.zeros((h, w, 1), dtype=np.int32)
        #img_pad = tf.cast(img_pad, tf.float32)
        #mask_pad = tf.cast(mask_pad, tf.int32)

        #print(h_new, w_new)

        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_pad[y:y+h_new, x:x+w_new, :] = mask

        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
        mask_crop = mask[y:y+h, x:x+w, :]
        #print(img_crop.dtype)
        #print(mask_crop.dtype)
        return mask_crop, img_crop

def copy_paste(mask_src, img_src, mask_main, img_main, lsj=True):

    mask_src, img_src = random_flip_horizontal(mask_src, img_src)
    mask_main, img_main = random_flip_horizontal(mask_main, img_main)

    # LSJ， Large_Scale_Jittering
    if lsj:
        mask_src, img_src = Large_Scale_Jittering(mask_src, img_src)
        mask_main, img_main = Large_Scale_Jittering(mask_main, img_main)
    else:
        # rescale mask_src/img_src to less than mask_main/img_main's size
        h, w, _ = img_main.shape
        mask_src, img_src = rescale_src(mask_src, img_src, h, w)

    img = img_add(img_src, img_main, mask_src)
    mask = img_add(mask_src, mask_main, mask_src)

    return mask, img





