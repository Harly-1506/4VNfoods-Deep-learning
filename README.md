# Classification and Segmentation 4 Vietnamese Foods

## Summary
- In this project, I use Pytorch to perform classification and classification of 4 popular dishes in Vietnam.Also, to make image segmentation easy, I used additional library [Segmentation Pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Dataset :egg: 
- The data is taken from the dataset [30VNFoods](https://www.kaggle.com/datasets/quandang/vietnamese-foods?fbclid=IwAR2bGtj0pe0SLybywrc5D-uS8ynXwqfDAZO6sTQ8eMLO7wcUP2wYCE4SJWw), this dataset includes 30 dishes and I took 4 dishes: Bánh mì, Phở, Bánh tráng nướng và Cơm tấm

  ![Example](https://github.com/Harly-1506/4VNfoods_Project/blob/main/images/image.png "This is a sample image.")

## Models
 - I use a variety of models ranging from MLP to Simple CNN, miniVGG. Pre-trained models such as VGG16, ResNet18

- For the Segmentation problem, I use the Unet structure with the encoders being pre-trained models to get the best results. 

- I use Wandb to track and compare experiments: [Classification](https://wandb.ai/harly/classifi_FoodVN?workspace=user-harly), [Segmentation](https://wandb.ai/harly/SegVNFood?workspace=user-harly)

## How to run this project :question:
```python
git clone https://github.com/Harly-1506/4VNfoods_Project.git

cd 4VNfoods_Project
#run classification
run classifi_main.py
#run segmentation
run seg_main.py
```
**__Note__**: When you run seg_main.py, it takes 8 to 10 minutes to prepare the data

## Result
![Example](https://github.com/Harly-1506/4VNfoods_Project/blob/main/images/W%26B%20valac.png "This is a sample image.")

## Demo:

- Demo program you can follow in this repository: [Demo](https://github.com/RC-Sho0/4VNFood--Demo-App-by-Streamlit)
___
*Author: Harly*

*If you have any problems, please leave a message in Issues*

*Give me a star :star: if you find it useful, thanks*
