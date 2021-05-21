# Martensite_Variants_Identification_Method_Graph

## Introduction
Martensite Variant Identification Method - Graph (MVIM-G) is a post-processing program using the function library in OVITO and Graph Neural Network. 

This method is developed by Yi-Ming Tseng, Pei-Te Wang, An-Cheng Yang, Nan-Yow Chen, and Prof. Nien-Ti Tsou*. It can help users to identify the microstructure of Nickel-Titanium alloys, such as Austenite, Orthorhombic, Monoclinic, and Rhombohedral.

The trained model, training dataset & testing dataset are available at the following link:
https://reurl.cc/o9NVmg

## Datasets
The dataset in the dataset_root folder has been pre-processed, the test example is an indentation simulation of Ni-Ti alloy at 325K. Besides, users also can use the PyTorch-geometric libraries to realize the composition of the dataset.

## Run the Code
### predict.py
Predict.py utilizes the pre-trained model to predict the martensite variants variant. In order to demonstrate, we choose the indentation case as a testing example, predict folder contains the raw data of the testing case. After executing the code, users can find the prediction data in the predict folder, and then users can observe the phase transformation in Ovito with the colormap.
