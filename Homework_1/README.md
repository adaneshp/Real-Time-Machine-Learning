# Problems Description:

## Problem 1:

1.a Take several pictures of red, blue, and green items with your phone or other digital cameras (or download some from the internet, if a camera isn’t available).

 

1.b Load each image, and convert it to a tensor.

 

1.c. For each image tensor, use the .mean() method to get a sense of how bright the image is.

 

1.d Take the mean of each channel of your images. Can you identify the red, green, and blue items from only the channel averages?

 

## Problem 2:

In our temperature prediction example, let’s change our model to a non-linear system. Consider the following description for our model:

 

w2 * t_u ** 2 + w1 * t_u + b.

 

2.a Modify the training loop properly to accommodate this redefinition. 

 

2.b Use 5000 epochs for your training. Explore different learning rates from 0.1 to 0.0001 (you need four separate trainings). Report your loss for every 500 epochs per training.

 

2.c Pick the best non-linear model and compare your final best loss against the linear model that we did during the lecture. For this, visualize the non-linear model against the linear model over the input dataset, as we did during the lecture. Is the actual result better or worse than our baseline linear model?

 

 

## Problem 3:

3.a. Develop preprocessing and a training loop to train a linear regression model that predicts housing price based on the following input variables:

 

area, bedrooms, bathrooms, stories, parking

 

For this, you need to use the housing dataset. See example on Canvas. Identify the best parameters for your linear regression model, based on the above input variables. In this case, you will have six parameters:

U=W5*X5 + W4*X4 + W3*X3 + W2*X2 + W1*X1 + B

 

3.b Use 5000 epochs for your training. Explore different learning rates from 0.1 to 0.0001 (you need four separate trainings). Report your loss for every 500 epochs per each training.

 

3.c Pick the best linear model (the one with the smaller final loss) and visualize it over the input dataset, as we did during the lecture.
