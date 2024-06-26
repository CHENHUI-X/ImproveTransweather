
# About this repo:

- Unofficial pytorch implementation for TransWeather -- CVPR 2022

- Original paper : [TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions](https://arxiv.org/abs/2111.14813)

- Note : Most of the code in this repository comes from the [original author's repository](https://github.com/jeya-maria-jose/TransWeather). Here, it is more about fine-tuning the source code, adding a lot of comments, fixing some existing bugs(or improve the performance), and adding some other details. 
(At the time of completing this repository, some details of the original author's code were not completed, so if the original author has an update, I will continue to follow up.)

- Plus, the original paper uses a larger dataset, which is inconvenient for people who just learn the model architecture and run the code successfully. So I sampled 600 images from the original dataset, 480 for training and 120 for testing, which you can download here:[Allweather_subset.zip](https://drive.google.com/file/d/1v1z7NRyF9wD6wAlZBbphBZgTuIs8zOas/view?usp=sharing), it's only 170mb.
---
# Introduction 

- Official web : [Website](https://jeya-maria-jose.github.io/transweather-web/)

- Official repo : [Repo](https://github.com/jeya-maria-jose/TransWeather)

- Model structure : <img src="/imgs/Transweather.png"  align= "center" />

- For details : <img src="/imgs/Transform_weather_structure _explain.png"  align= "center" />
