# Classification and Segmentation 4 Vietnamese Foods

## Summary
- In this project, I use Pytorch to perform classification and segmentation of 4 popular dishes in Vietnam.Also, to make image segmentation easy, I used additional library [Segmentation Pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Dataset :egg: 
- The data is taken from the dataset [30VNFoods](https://www.kaggle.com/datasets/quandang/vietnamese-foods?fbclid=IwAR2bGtj0pe0SLybywrc5D-uS8ynXwqfDAZO6sTQ8eMLO7wcUP2wYCE4SJWw), this dataset includes 30 dishes and I took 4 dishes: Bánh mì, Phở, Bánh tráng nướng và Cơm tấm

  ![Example](https://github.com/Harly-1506/4VNfoods_Project/blob/main/images/image.png "This is a sample image.")

## Models
 - I use a variety of models ranging from MLP to Simple CNN, miniVGG. Pre-trained models such as VGG16, ResNet18

- For the Segmentation problem, I use the Unet structure with the encoders being pre-trained models to get the best results. 

- I use Wandb to track and compare experiments: [Classification](https://wandb.ai/harly/classifi_FoodVN?workspace=user-harly), [Segmentation](https://wandb.ai/harly/SegVNFood?workspace=user-harly)

## How to run this project :question:
```python
git clone https://github.com/Harly-1506/4VNfoods-Deep-learning.git

cd 4VNfoods_Deep_learning
#run classification
run classifi_main.py
#run segmentation
run seg_main.py
```
**__Note__**: When you run seg_main.py, it takes 8 to 10 minutes to prepare the data
## Classification Results
|     Methods                |     Accuracy    |     Loss        |     Val_Accuracy    |     Val_Loss    |     Test_accuracy    |
|----------------------------|-----------------|-----------------|---------------------|-----------------|----------------------|
|     Resnet18_pretrained    |     99.926      |     6.78E-05    |     96.907          |     0.1106      |     95.886           |
|     Resnet18               |     99.486      |     0.0003      |     80.154          |     0.7141      |     78.663           |
|     VGG16_pretrained       |     99.266      |     0.0005      |     94.587          |     0.4035      |     95.758           |
|     VGG16                  |     95.229      |     0.0030      |     78.350          |     0.6939      |     77.763           |
|     miniVGG                |     99.926      |     0.0001      |     82.989          |     0.6325      |     87.917           |
|     SimpleCNN              |     99.559      |     0.0008      |     86.597          |     0.3855      |     86.632           |
|     MLP_4hidden512node     |     53.651      |     0.0678      |     45.103          |     2.8904      |     47.043           |
|     MLP_3hidden1024node    |     44.403      |     0.1080      |     34.278          |     4.8297      |     38.946           |
|     MLP_3hidden512node     |     55.486      |     0.0707      |     40.721          |     5.5563      |     44.987           |
|     MLP_4hidden            |     47.706      |     0.0583      |     37.886          |     2.3706      |     38.303           |
|     MLP_3hidden            |     49.761      |     0.0512      |     36.082          |     3.0187      |     41.902           |
|     MLP_2hidden            |     48.844      |     0.0438      |     40.979          |     1.6916      |     41.516           |
## Segmentation Results
|     Methods          |     iou/valid    |     iou     banhmi    |     iou     banhtrang    |     iou     comtam    |     iou     pho    |     iou_clutter    |
|----------------------|------------------|-----------------------|--------------------------|-----------------------|--------------------|--------------------|
|     Unet_ResNet34    |     0.8625       |     0.8273            |     0.8529               |     0.7083            |     0.7099         |     0.9084         |
|     Unet-ResNet18    |     0.8828       |     0.8655            |     0.8897               |     0.7893            |     0.7571         |     0.9214         |
|     Unet-VGG16       |     0.8716       |     0.8627            |     0.8713               |     0.7395            |     0.7463         |     0.9146         |
## Plot Val Accuracy
- Classification:
![Example](https://github.com/Harly-1506/4VNfoods_Project/blob/main/images/W%26B%20valac.png "This is a sample image.")
- Segmentation:
![image](https://github.com/Harly-1506/4VNfoods-Deep-learning/assets/86733695/6d772489-a7a4-47b6-b6e9-5fe7da503fd3)

## Demo:

- Demo program you can follow in this repository: [Demo](https://github.com/RC-Sho0/4VNFood--Demo-App-by-Streamlit)
___
*Author: Harly*

*If you have any problems, please leave a message in Issues*

*Give me a star :star: if you find it useful, thanks*
