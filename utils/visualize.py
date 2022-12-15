import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import tqdm
import numpy as np

def visualize(image, mask, label=None, truth=None,  augment=False):
    if truth is None:
        plt.figure(figsize=(14, 20))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        if augment == False:
            plt.title(f"{'Original Image'}")
        else:
            plt.title(f"{'Augmented Image'}")

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        if label is not None:
            plt.title(f"{label.capitalize()}")
        else:
            plt.title(f"{'Ground Truth'}")

    else:
        plt.figure(figsize=(26, 36))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"{'Original Image'}")

        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.title(f"{'Prediction'}")
        
        plt.subplot(1, 3, 3)
        plt.imshow(truth)
        plt.title(f"{'Ground Truth'}")
        
def visualize_overlay(image, mask, truth_path=None):
    if truth_path is None:
        plt.figure(figsize=(26, 36))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"{'Original Image'}")

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title(f"{'Prediction'}")
        
    else:
        truth = Image.open(truth_path)
        truth = truth.resize((224, 224), Image.ANTIALIAS)
        plt.figure(figsize=(26, 36))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"{'Original Image'}")

        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.title(f"{'Prediction'}")
        
        plt.subplot(1, 3, 3)
        plt.imshow(truth)
        plt.title(f"{'Ground Truth'}")
        
def visualize_prediction(image, mask):
        plt.figure(figsize=(26, 36))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"{'Original Image'}")

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title(f"{'Prediction'}")



def plot_cm(model, testloader):

  y_true = []
  y_pred = []
  
  for i, (images, labels) in tqdm.tqdm(
        enumerate(testloader), total = len(testloader), leave = True, colour = "blue", desc = f"Testing",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):
      images = images.cuda(non_blocking = True)
      labels = labels.cuda(non_blocking = True)
        
      y_true.extend(labels.cpu().numpy())
    
      outputs=model(images)
    
      _, predicted = outputs.max(1)
      y_pred.extend(predicted.cpu().numpy())

    
  cf_matrix = confusion_matrix(y_true, y_pred)
  cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
  class_names = ( "Bánh mì","Cơm tấm","Phở","Bánh tráng nướng")
  
  # Create pandas dataframe
  dataframe = pd.DataFrame(cmn, index=class_names, columns=class_names)
  plt.figure(figsize=(8, 6))
  
  # Create heatmap
  sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt=".2f")
  
  plt.title("Confusion Matrix"), plt.tight_layout()
  
  plt.ylabel("True Class"), 
  plt.xlabel("Predicted Class")
  plt.show()