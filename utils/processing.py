import os
from torchvision import transforms, utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import cv2
import re
from tqdm import tqdm
from PIL import Image
import random
import os.path as osp
from typing import Callable, List, Tuple

root_train = "/content/4VNfoods_Project/datasets/dataset/Train"
root_val = "/content/4VNfoods_Project/datasets/dataset/Val"
root_test =  "/content/4VNfoods_Project/datasets/dataset/Test"


Name_food = {
    0:"Banh mi",
    1:"Banh Trang",
    2:"Com tam",
    3:"Pho"
}

#augmentation
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomGrayscale(p = 0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    # A.RandomBrightnessContrast(p=0.2),
    
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_seg= transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    
])
test_seg = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

import albumentations as albu

def get_training_augmentation():
    train_transform = [

        albu.Resize(224, 224, p=1),
        albu.HorizontalFlip(p=0.5),

        albu.OneOf([
            albu.RandomBrightnessContrast(
                  brightness_limit=0.4, contrast_limit=0.4, p=1),
            albu.CLAHE(p=1),
            albu.HueSaturationValue(p=1)
            ],
            p=0.9,
        ),

        albu.IAAAdditiveGaussianNoise(p=0.2),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [albu.Resize(224, 224, p=1),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
    


def get_path_images_labels(path):
  images = []
  labels = []
  for label in os.listdir(path):
    for image in os.listdir(path + "/" + label):
      images.append(path + "/" + label + "/" + image)
      labels.append(label)
  return images, labels


def getAllDataset():
  train_paths, train_labels = get_path_images_labels(root_train)
  val_paths, val_labels = get_path_images_labels(root_val)
  test_paths, test_labels = get_path_images_labels(root_test)

  lb = LabelEncoder()
  train_labels = lb.fit_transform(train_labels)
  val_labels = lb.fit_transform(val_labels)
  test_labels = lb.fit_transform(test_labels)

  return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def plot_images(train_loader):
  samples, labels = iter(train_loader).next()

  fig = plt.figure(figsize=(16,24))
  for i in range(24):
      a = fig.add_subplot(4,6,i+1)
      a.set_title(Name_food[labels[i].item()])
      a.axis('off')
      a.imshow(np.transpose(samples[i].numpy(), (1,2,0)))
  plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)


  # processing for segmentaion


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

# def getallImages(images_paths, mask_paths):
#   images = []
#   masks = []
#   for i in range(len(images_paths)):
    
def get_paths_seg_img_mask(path):
  images = []
  masks = []
  root = "/content/4VNFood"
  foods = os.listdir(root)
  for food in foods:
    paths_images= root + "/" + food + "/" + "images"
    paths_mask = root + "/" + food + "/" + "labels"
    allpaths = os.listdir(paths_images)
    allmask = os.listdir(paths_mask)
    allpaths.sort(key = natural_keys)
    allmask.sort(key = natural_keys)
    for i in range(len(allpaths)):
      images.append(paths_images + "/" + allpaths[i])
      masks.append(paths_mask + "/" + allmask[i])
  return images, masks

def get_paths_seg_img_mask1(path):
  images = []
  masks = []
  root = "/content/4VNFood"
  root1 = "/content/4VNFoodNew"
  foods = os.listdir(root)
  for food in foods:
    paths_images= root + "/" + food + "/" + "images"
    paths_mask = root1 + "/" + food + "/" + "TrainId"
    allpaths = os.listdir(paths_images)
    allmask = os.listdir(paths_mask)
    allpaths.sort(key = natural_keys)
    allmask.sort(key = natural_keys)
    for i in range(len(allpaths)):
      images.append(paths_images + "/" + allpaths[i])
      masks.append(paths_mask + "/" + allmask[i])
  return images, masks

class UAVidColorTransformer:
    def __init__(self):
    # color table.
        self.clr_tab = self.createColorTable()
    # id table.
        id_tab = {}
        for k, v in self.clr_tab.items():
            id_tab[k] = self.clr2id(v)
        self.id_tab = id_tab

    def createColorTable(self):
        clr_tab = {}
        clr_tab['Clutter'] = [0, 0, 0]
        clr_tab['BanhMi'] = [236, 176, 31]
        clr_tab['Pho'] = [0, 113, 188]
        clr_tab['ComTam'] = [125 ,46, 141]
        clr_tab['BanhTrang'] = [118, 171 , 47]
        return clr_tab

    def colorTable(self):
        return self.clr_tab
   
    def clr2id(self, clr):
        return clr[0]+clr[1]*255+clr[2]*255*255

  #transform to uint8 integer label
    def transform(self,label, dtype=np.int32):
        height,width = label.shape[:2]
    # default value is index of clutter.
        newLabel = np.zeros((height, width), dtype=dtype)
        id_label = label.astype(np.int64)
        id_label = id_label[:,:,0]+id_label[:,:,1]*255+id_label[:,:,2]*255*255
        for tid,val in enumerate(self.id_tab.values()):
            mask = (id_label == val)
            newLabel[mask] = tid
        return newLabel

  #transform back to 3 channels uint8 label
    def inverse_transform(self, label):
        label_img = np.zeros(shape=(label.shape[0], label.shape[1],3),dtype=np.uint8)
        values = list(self.clr_tab.values())
        for tid,val in enumerate(values):
            mask = (label==tid)
            label_img[mask] = val
        return label_img
        
clrEnc = UAVidColorTransformer()

def prepareTrainIDForDir(gtDirPath, saveDirPath):
    gt_paths = ["banhmi", "banhtrang", "comtam", "pho"]
    for pd in tqdm(gt_paths):
        lbl_dir = osp.join(gtDirPath, pd, 'labels')
        lbl_paths = os.listdir(lbl_dir)
        if not osp.isdir(osp.join(saveDirPath, pd, 'TrainId')):
            os.makedirs(osp.join(saveDirPath, pd, 'TrainId'))
            assert osp.isdir(osp.join(saveDirPath, pd, 'TrainId')), 'Fail to create directory:%s'%(osp.join(saveDirPath, pd, 'TrainId'))
        for lbl_p in lbl_paths:
            lbl_path = osp.abspath(osp.join(lbl_dir, lbl_p))
            trainId_path = osp.join(saveDirPath, pd, 'TrainId', lbl_p)
            gt = np.array(Image.open(lbl_path))
            trainId = clrEnc.transform(gt, dtype=np.uint8)
            Image.fromarray(trainId).save(trainId_path)
