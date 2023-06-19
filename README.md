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
|    <br>Methods            |    <br>Accuracy    |    <br>Loss        |    <br>Val_Accuracy    |    <br>Val_Loss    |    <br>Test_accuracy    |
|-------------------------------|--------------------|--------------------|------------------------|--------------------|-------------------------|
|    <br>Resnet18_pretrained    |    <br>99.926      |    <br>6.78E-05    |    <br>96.907          |    <br>0.1106      |    <br>95.886           |
|    <br>Resnet18               |    <br>99.486      |    <br>0.0003      |    <br>80.154          |    <br>0.7141      |    <br>78.663           |
|    <br>VGG16_pretrained       |    <br>99.266      |    <br>0.0005      |    <br>94.587          |    <br>0.4035      |    <br>95.758           |
|    <br>VGG16                  |    <br>95.229      |    <br>0.0030      |    <br>78.350          |    <br>0.6939      |    <br>77.763           |
|    <br>miniVGG                |    <br>99.926      |    <br>0.0001      |    <br>82.989          |    <br>0.6325      |    <br>87.917           |
|    <br>SimpleCNN              |    <br>99.559      |    <br>0.0008      |    <br>86.597          |    <br>0.3855      |    <br>86.632           |
|    <br>MLP_4hidden512node     |    <br>53.651      |    <br>0.0678      |    <br>45.103          |    <br>2.8904      |    <br>47.043           |
|    <br>MLP_3hidden1024node    |    <br>44.403      |    <br>0.1080      |    <br>34.278          |    <br>4.8297      |    <br>38.946           |
|    <br>MLP_3hidden512node     |    <br>55.486      |    <br>0.0707      |    <br>40.721          |    <br>5.5563      |    <br>44.987           |
|    <br>MLP_4hidden            |    <br>47.706      |    <br>0.0583      |    <br>37.886          |    <br>2.3706      |    <br>38.303           |
|    <br>MLP_3hidden            |    <br>49.761      |    <br>0.0512      |    <br>36.082          |    <br>3.0187      |    <br>41.902           |
|    <br>MLP_2hidden            |    <br>48.844      |    <br>0.0438      |    <br>40.979          |    <br>1.6916      |    <br>41.516           |
## Segmentation Results
|    <br>Methods          |    <br>iou/valid    |    <br>iou<br>   <br>banhmi    |    <br>iou<br>   <br>banhtrang    |    <br>iou<br>   <br>comtam    |    <br>iou<br>   <br>pho    |    <br>iou_clutter    |
|-------------------------|---------------------|--------------------------------|-----------------------------------|--------------------------------|-----------------------------|-----------------------|
|    <br>Unet_ResNet34    |    <br>0.8625       |    <br>0.8273                  |    <br>0.8529                     |    <br>0.7083                  |    <br>0.7099               |    <br>0.9084         |
|    <br>Unet-ResNet18    |    <br>0.8828       |    <br>0.8655                  |    <br>0.8897                     |    <br>0.7893                  |    <br>0.7571               |    <br>0.9214         |
|    <br>Unet-VGG16       |    <br>0.8716       |    <br>0.8627                  |    <br>0.8713                     |    <br>0.7395                  |    <br>0.7463               |    <br>0.9146         |
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
