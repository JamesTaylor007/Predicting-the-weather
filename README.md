# Predicting-the-weather
As part of my second year of university we had to write an artificial neural network to try and predict the pan evaporation of the next year haven been given the previous 3 years of weather data. This data set included: Mean daily temperature, wind speed, solar radiation, air pressure, humidity, Panevapouration. The goal was to use the first 5 of these to then predicit the pan evapouration. 

# Cleaning the data

The data that was provided to us had a few spurious data points, therefore the first step was to clean this data. The approach I took to do this was to plot the data points against the dates when they occured and then you should be able to spot any which deviate from values that would you expect. Ie:

<img width="485" alt="Screen Shot 2021-08-02 at 10 44 53" src="https://user-images.githubusercontent.com/62481908/127841402-030c9e28-f76f-4a82-8c63-3898336276ad.png">

As you can see in thr graph above straight away there are points which you know to be inncorrect. The hottest day ever recorded is 56.7 degrees and therefore you can automatically disregard a few of those data points. I then repeated this process for every predictor and predictand. 

# Splitting up the data

We need to split the data up into 3 sets:

- A set of training data used by our training algorithm.
- A set of validation data to ensure that our algorithm is not just useful on the training
data and nothing but the training data.
- A set of test data so that we can check the algorithm is working as intended and get an idea of how accurately our model is predicting the pan-evapouration.


# Learning Algorithm

Step 1: Initialise weights and biases to random starting points.

- Choose a small step size parameter, (ie 0.1)
- Assign random small weights and biases to all nodes

Step 2: Select a data point
Step 3: Make a forward pass and then calculate error

- Make a forward pass through the network computing the weighted sums (Sj)
and the activation Uj=f(sj) for every node where f(sj) is the activation function.

Step 4: Make a backward pass adjusting weights according to the error

- (delta)j = (Cj - Uj)f’(Sj) if j is an output node where:
  - Cj is the correct output
  - Uj is the output you have
  - f’(Sj) is the differential of the activation function
  - Sj is the weighted sums
  - (delta)j is the output weight
 
- (delta)j = ( ΣWj, m δm )f’(Sj) for other nodes
  - Wi input weight of the node
  - Wj is the output weight of the node
  - (delta)m is the weight of the output node
  - (delta)j
  
Step 5: Repeat

Here is a diagram I pulled off google of a neural network for working out the magnitude of earthquakes. We are doing the exact same thing but trying to do it for the pan-evapouration:

![weighted-artificial-neural-network](https://user-images.githubusercontent.com/62481908/127844821-965d84ab-9783-4162-8928-fc28be336923.jpg)

# Results 
Below is a plot of the predictions my model made (in orange) against the acctual figures for that year (in blue). You can see straight away that it is not perfect however it doesn't look too far off! 

<img width="485" alt="Screen Shot 2021-08-02 at 11 09 24" src="https://user-images.githubusercontent.com/62481908/127844938-16110404-cd7d-4191-baaa-fa67c89b5b99.png">

