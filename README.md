# Occupancy Networks for Single Image Reconstruction
![Example 1](img/00.gif)
![Example 2](img/01.gif)
![Example 3](img/02.gif)

This repository contains the code for our project for the course 'Machine Learning for 3D Geometry' at TUM [IN2392]. In this work, we had improved the performance of the model in terms of Intersection over Union (IoU) compared to the baseline [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks). Additionally, we had reduced the number of parameters to make the training of the model more efficient.


<strong>Authors</strong>
1. Yujun Lin
2. Joong-Won Seo
3. Yunan Li



## Dataset

For our experiments, we use [Pix3D](https://github.com/xingyuansun/pix3d) as our new dataset which contains images, masks, meshes and camera positions.

## Modifications
1. Replace the backbone ResNet with ConvNeXt
2. Integrate feature pyramid to enable multi-scale inputs
3. Incorporate camera pose information

## Results
To see the experiment results, please check the final paper.
