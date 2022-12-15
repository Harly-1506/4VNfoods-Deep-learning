import os 


# lib_dir = "utils/segment_ai/" 

# os.system(f"cp -r {lib_dir} /content/segment_ai")

from utils.segment_ai.get_data import *
path = "segments/hoangsonvothanh_4VNFood/v1"
new_path = "/content/4VNFood" #Doi ten thu muc neu muon
clone_data()
extract2fol(path,new_path)
os.system("rm -rf /content/segments")
#output_type="semantic"


import os
os.system("pip install wandb")
os.system("pip install segmentation-models-pytorch")
os.system("pip install catalyst==20.07")



from torch.utils.data import Dataset, DataLoader
import PIL

import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split

from utils.processing import *
from utils.vnfood_ds import *
from utils.segfood_ds import *
from utils.trainer import *
from utils.visualize import *
from model.mlp import *
from model.cnn import *
from model.vggnet import *
from model.resnet import *
from model.unet import UNet
from segmentation_models_pytorch.losses import DiceLoss


import random
import matplotlib.pyplot as plt
import numpy as np
from catalyst import utils
import cv2
import glob
from PIL import Image
import os.path as osp
from tqdm import tqdm
from typing import Callable, List, Tuple
import torch
import catalyst
import wandb

#check your path before running
prepareTrainIDForDir('/content/4VNFood', '/content/4VNFoodNew')

images, mask = get_paths_seg_img_mask1("paths")

X_train, X_test, y_train, y_test = train_test_split(images,mask , test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test,y_test , test_size=0.5, random_state=42)

import segmentation_models_pytorch as smp

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
labels = ['clutter',"banhmi", "banhtrang", "comtam", "pho"]

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(labels), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

labels = ['clutter',"banhmi", "banhtrang", "comtam", "pho"]
train_dataset = segDataset(
    X_train, 
    y_train, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=labels,
)

valid_dataset = segDataset(
    X_val, 
    y_val, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=labels,
)
test_dataset = segDataset(
    X_test, 
    y_test, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=labels,
)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers = 2, pin_memory= True)
valid_loader = DataLoader(valid_dataset, batch_size=12, shuffle=False, num_workers = 2, pin_memory= True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 2, pin_memory= True)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

labels = ['clutter',"banhmi", "banhtrang", "comtam", "pho"]
for i in range(10):
    dataset = segDataset(X_train, y_train)

    image, mask = dataset[i]
    visualize(
        image=image, mask=mask.squeeze())

from catalyst.contrib.nn import BCEDiceLoss, Adam, Lookahead, OneCycleLRWithWarmup
from catalyst.dl import SupervisedRunner
# from catalyst.contrib.nn import Adam, , OneCycleLRWithWarmup
from catalyst.dl import SupervisedRunner

logdir = "/content/logs"
num_epochs = 15
learning_rate = 1e-3
base_optimizer = Adam([
    {'params': model.parameters(), 'lr': learning_rate}, 
])
optimizer = Lookahead(base_optimizer)
criterion = BCEDiceLoss(activation=None)
runner = SupervisedRunner()
scheduler = OneCycleLRWithWarmup(
    optimizer, 
    num_steps=num_epochs, 
    lr_range=(0.0016, 0.0000001),
    init_lr = learning_rate,
    warmup_steps=2
)

from catalyst.dl.callbacks import IouCallback, WandbLogger, EarlyStoppingCallback, ClasswiseIouCallback

callbacks = [
    IouCallback(activation = 'none'),
    ClasswiseIouCallback(classes=labels, activation = 'none'),
    # EarlyStoppingCallback(patience=7, metric='iou', minimize=False),
    # WandbLogger(project='SegVNFood', name='Unet_ResNet34'),
    
]

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=callbacks,
    logdir=logdir,
    num_epochs=num_epochs,
    # save our best checkpoint by IoU metric
    main_metric="iou",
    # IoU needs to be maximized.
    minimize_metric=False,
    # for FP16. It uses the variable from the very first cell
    # fp16=fp16_params,
    # prints train logs
    verbose=True,
)

# create test dataset
test_dataset = segDataset(
    X_test, 
    y_test, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=labels,
)

infer_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

# this get predictions for the whole loader
predictions = np.vstack(list(map(
    lambda x: x["logits"].cpu().numpy(), 
    runner.predict_loader(loader=infer_loader, resume=f"{logdir}/checkpoints/best.pth")
)))

print(type(predictions))
print(predictions.shape)

from catalyst.utils import mask_to_overlay_image

threshold = 0.5
max_count = 10
test_dataset = segDataset(
    X_test, 
    y_test, 
    augmentation=get_validation_augmentation(), 
    # preprocessing=get_preprocessing(preprocessing_fn),
    classes=labels,
)
for i, ((features,gr_true), logits) in enumerate(zip(test_dataset, predictions)):
    image = features
    # print(image.shape)
    # image = (image * 255).astype(np.uint8)
    # mask_ = torch.from_numpy(logits[0]).sigmoid()
    # mask = utils.detach(mask_ > threshold).astype("float")
    # image =np.moveaxis(image, 0, -1)
    # mask =np.moveaxis(mask, 0, -1)
    # show_examples(name ="",image=image, mask=mask)
    mask = (predictions[i])
    # mask = np.moveaxis(mask, 0, -1 )
    pred = mask_to_overlay_image(image=image, masks=mask ,threshold=0.4)
    # pred = clrEnc.inverse_transform(pred.squeeze())
    visualize_overlay(image, pred, truth_path = y_test[i])
    
    if i >= max_count:
        break

