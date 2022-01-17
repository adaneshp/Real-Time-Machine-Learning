Homework 0

Instructions

Problem 1 (20pts):

Pass five random images (from the internet) to ResNet 101, and analyze the outcomes. Analyze and report top1 and top 5 accuracies across the five images. Report any anomaly or miss -classification you observe. Try to come with logical reasoning (educated guesses) for miss-prediction. Make sure to perform proper pre-processing and adjustment and normalization before passing the images. Make sure to include images in the package you submit.

 

Problem 2 (20pts):

Pass five random images containing horses (one and multiple houses) to ResnetGen network and analyze the outcomes. Make sure to include both input images and generated images in your report. Try to come with logical reasoning (educated guesses) for miss-generations. Make sure to perform proper pre-processing and adjustment and normalization before passing the images.

 

Problem 3 (20pts):

Ptflops is a great tool to calculate the computational complexity of algorithms (number of MACs, and model size). This script is designed to compute the theoretical number of multiply-add operations in convolutional neural networks. It also can compute the number of parameters and print the computational cost (Number of MACs) of a given network. Here is the link to the tool:

https://pypi.org/project/ptflops/ (Links to an external site.)

use Ptflops and report the number of MACs and models size for both Problem 1 and Problem 2 over the batch size of 1 (running only one image at a time). 

 

Problem 4 (40pts): 

The MobileNet v2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input. MobileNet v2 uses lightweight depth-wise convolutions to filter features in the intermediate expansion layer. Additionally, non-linearities in the narrow layers were removed in order to maintain representational power.

The pre-trained model is part of torchvision. Here is the link to MobileNet v2:
https://pytorch.org/hub/pytorch_vision_mobilenet_v2/ (Links to an external site.)

 

Like problem 1, Analyze and report top1 and top 5 accuracies across the five images. Report any anomaly or miss -classification you observe. Try to come with logical reasoning (educated guesses) for miss-prediction. Make sure to perform proper pre-processing and adjustment and normalization before passing the images. Make sure to include images in the package you submit.

 

Like problem 3, use Ptflops and report the number of MACs and models size for both Problem 1 and Problem 2 over the batch size of 1 (running only one image at a time). 
