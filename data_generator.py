"""
每个batch进行不同的数据增强
"""

import os
import random
import numpy as np
import pickle
import sys
sys.path.append(os.path.split(os.getcwd())[0])
sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
from augmentation import augment
from keras.utils import to_categorical

each_batchsize = 16
task = "tumor_yz" # astr_epen MS_NMO
train_choices, val_choices = [True], [False]
features = "t2+t1c" # "t1c", "t2", "t1c+t2"
process_method = "zscore_clip" # zscore_no_clip zscore_clip minmax
delete_indexs = []

split_path = "../split_task.pkl".replace("task", task)
pkl_root = "/hd/Lizh/Lizh/spinal_classification_t2_t1c/pkls"
pkl_path = os.path.join(pkl_root, process_method)


with open(split_path, "rb") as f:
    split = pickle.load(f)
pkls = os.listdir(pkl_path)

train0_filedirs, train1_filedirs = split["train0"], split["train1"]
pkls0_train = sorted([i for i in pkls if i.split("level")[0] in train0_filedirs])
pkls1_train = sorted([i for i in pkls if i.split("level")[0] in train1_filedirs])
random.shuffle(pkls0_train)
random.shuffle(pkls1_train)
train_0_batches = len(pkls0_train) // each_batchsize
train_1_batches = len(pkls1_train) // each_batchsize
train_batches = max(train_0_batches, train_1_batches)

def train_0_generator():
    while True:
        for i in range(train_0_batches):
            batchx = [] # (each_batchsize, 256, 256, 3)
            batchy = [] # (each_batchsize, 2)
            batch_pkls0 = pkls0_train[i*each_batchsize:(i+1)*each_batchsize]
            for file in batch_pkls0:
                with open(os.path.join(pkl_path, file), "rb") as f:
                    onex = pickle.load(f) # (256, 256, 3)
                image, mask = onex[:, :, 0:2], onex[:, :, 2:]
                image, mask = np.moveaxis(image, -1, 0), np.moveaxis(mask, -1, 0)
                image, mask = augment(image, mask, choices=train_choices)
                image, mask = np.moveaxis(image, 0, -1), np.moveaxis(mask, 0, -1)
                image = np.concatenate([image, mask], axis=-1)

                for index in delete_indexs:
                    image[:, :, index] = 0

                batchx.append(image) # (256, 256, 3)
                batchy.append([1, 0])
            batchx, batchy = np.array(batchx), np.array(batchy)
            yield batchx, batchy
        random.shuffle(pkls0_train)

def train_1_generator():
    while True:
        for i in range(train_1_batches):
            batchx = [] # (each_batchsize, 256, 256, 3)
            batchy = [] # (each_batchsize, 2)
            batch_pkls1 = pkls1_train[i*each_batchsize:(i+1)*each_batchsize]
            for file in batch_pkls1:
                with open(os.path.join(pkl_path, file), "rb") as f:
                    onex = pickle.load(f) # (256, 256, 3)
                image, mask = onex[:, :, 0:2], onex[:, :, 2:]
                image, mask = np.moveaxis(image, -1, 0), np.moveaxis(mask, -1, 0)
                image, mask = augment(image, mask, choices=train_choices)
                image, mask = np.moveaxis(image, 0, -1), np.moveaxis(mask, 0, -1)
                image = np.concatenate([image, mask], axis=-1)

                for index in delete_indexs:
                    image[:, :, index] = 0

                batchx.append(image) # (256, 256, 3)
                batchy.append([0, 1])
            batchx, batchy = np.array(batchx), np.array(batchy)
            yield batchx, batchy
        random.shuffle(pkls1_train)

def train_generator():
    train_0_gen = train_0_generator()
    train_1_gen = train_1_generator()
    while True:
        batchx0, batchy0 = next(train_0_gen)
        batchx1, batchy1 = next(train_1_gen)
        batchx = np.concatenate([batchx0, batchx1], axis=0) # (batchsize, 256, 256, 3)
        batchy = np.concatenate([batchy0, batchy1], axis=0) # (batchsize, 2)
        yield batchx, batchy

val0_filedirs, val1_filedirs = split["val0"], split["val1"]
pkls0_val = sorted([i for i in pkls if i.split("level")[0] in val0_filedirs])
pkls1_val = sorted([i for i in pkls if i.split("level")[0] in val1_filedirs])
random.shuffle(pkls0_val)
random.shuffle(pkls1_val)
val_0_batches = len(pkls0_val) // each_batchsize
val_1_batches = len(pkls1_val) // each_batchsize
val_batches = max(val_0_batches, val_1_batches)

def val_0_generator():
    while True:
        for i in range(val_0_batches):
            batchx = [] # (each_batchsize, 256, 256, 3)
            batchy = [] # (each_batchsize, 2)
            batch_pkls0 = pkls0_val[i*each_batchsize:(i+1)*each_batchsize]
            for file in batch_pkls0:
                with open(os.path.join(pkl_path, file), "rb") as f:
                    onex = pickle.load(f) # (256, 256, 3)
                image, mask = onex[:, :, 0:2], onex[:, :, 2:]
                image, mask = np.moveaxis(image, -1, 0), np.moveaxis(mask, -1, 0)
                image, mask = augment(image, mask, choices=val_choices)
                image, mask = np.moveaxis(image, 0, -1), np.moveaxis(mask, 0, -1)
                image = np.concatenate([image, mask], axis=-1)

                for index in delete_indexs:
                    image[:, :, index] = 0

                batchx.append(image) # (256, 256, 3)
                batchy.append([1, 0])
            batchx, batchy = np.array(batchx), np.array(batchy)
            yield batchx, batchy
        random.shuffle(pkls0_val)

def val_1_generator():
    while True:
        for i in range(val_1_batches):
            batchx = [] # (each_batchsize, 256, 256, 3)
            batchy = [] # (each_batchsize, 2)
            batch_pkls1 = pkls1_val[i*each_batchsize:(i+1)*each_batchsize]
            for file in batch_pkls1:
                with open(os.path.join(pkl_path, file), "rb") as f:
                    onex = pickle.load(f) # (256, 256, 3)
                image, mask = onex[:, :, 0:2], onex[:, :, 2:]
                image, mask = np.moveaxis(image, -1, 0), np.moveaxis(mask, -1, 0)
                image, mask = augment(image, mask, choices=val_choices)
                image, mask = np.moveaxis(image, 0, -1), np.moveaxis(mask, 0, -1)
                image = np.concatenate([image, mask], axis=-1)

                for index in delete_indexs:
                    image[:, :, index] = 0

                batchx.append(image) # (256, 256, 3)
                batchy.append([0, 1])
            batchx, batchy = np.array(batchx), np.array(batchy)
            yield batchx, batchy
        random.shuffle(pkls1_val)

def val_generator():
    val_0_gen = val_0_generator()
    val_1_gen = val_1_generator()
    while True:
        batchx0, batchy0 = next(val_0_gen)
        batchx1, batchy1 = next(val_1_gen)
        batchx = np.concatenate([batchx0, batchx1], axis=0) # (batchsize, 256, 256, 3)
        batchy = np.concatenate([batchy0, batchy1], axis=0) # (batchsize, 2)
        yield batchx, batchy
