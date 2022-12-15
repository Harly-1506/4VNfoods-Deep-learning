from torch import is_anomaly_enabled
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import PIL
import torch
import cv2
from skimage.io import imread, imsave

# seg= transforms.Compose([
#     transforms.ToTensor(),
    
# ])


class FoodVNDs(Dataset):
  def __init__(self, image_paths, labels, transform = None):
    self.image_paths = image_paths
    self.labels = labels
    self.transform = transform
    

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image = PIL.Image.open(self.image_paths[idx])
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image)

    return image, label
  


class segFoodVNDs(Dataset):
  def __init__(self, image_paths, image_masks, transform = None):
    self.image_paths = image_paths
    self.image_masks = image_masks
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)
  
  def __getitem__(self, idx):
    image = np.array(PIL.Image.open(self.image_paths[idx]).convert('RGB'))
    mask = np.array(PIL.Image.open(self.image_masks[idx]).convert('RGB'), dtype=np.float64)
    # image = imread(self.image_paths[idx])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = imread(self.image_masks[idx])
    # mask = cv2.cvtColor(mask,  cv2.COLOR_BGR2RGB)
    # print(image.shape)
    # mask  = np.moveaxis(mask, -1, 0)    
    # mask = mask / 255.0
    # mask = (mask*255).astype(np.uint8)
    # mask = mask[:,:,1]
    # mask = mask = mask.astype('long')


    # mask[mask > 0] = 1
    # 
    if self.transform:
      aug = self.transform(image=image, mask = mask)
      image = PIL.Image.fromarray(aug['image'])
      mask = aug['mask']
    
    t = transforms.Compose([transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    image = t(image)
    # mask = t(mask)
    mask = mask.astype(np.int_)/255
    # mask = torch.from_numpy(mask).long()
    mask = mask[:,:,1]
    # mask = mask.unsqueeze(0)
    # mask = mask/255.0

 
    # print(image.shape)
    # print(mask)
    return image, mask
