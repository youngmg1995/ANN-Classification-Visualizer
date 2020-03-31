# ANN-Classification-Visualizer
Interactive GUI for representing how neural networks classify clusters using of data

## Description
The purpose of this project was to build an visual representation of a neural networks process for classifying datasets. In doing we we can gain a better intuitve understanding of how neural networks classify data. To accomplish this task we we will classify a simple set of 2-D data points: (x,y) into two groups using dense neural network with structure 2-50-50-2. As the network is trained, we will visulaize the process by displaying the regions for each classification overtop of our dataset.

## Instructions for Use
1) Run file "Classification_Visualizer.py" using python console, from command line, etc., to initiate visualizer window.
2) Use cursor to select group of data points from left figure. To do so left click over the figure and hold mouse while drawing a closed or nearly closed loop around desired points. Let go of mouse to finish drawing.
3) Press "Enter" key to accept these points and initate the training process for our neural network classifier.
4) Observe the training process and classification regions on the right window. Status will indicate "Finished" when all the points have been properly classified by the model.
5) Repeat steps 1-4 to classify different selections from our dataset or close the window if finished.
5) (Optional) If the network is having trouble training and classifying a particular selection, the user may at any point press the "Backspace" key to cancel the training. Then repeat steps 1-4 to classify a new selection or close the window if finished.

#### Note on Use
The neural network can easily classify most simple shapes and datapoint selections in a few steps. However the model will take longer the more complex and disjoint the selection becomes. In particular, the network has the most trouble when points close together and placed in the different groups (selected and unselected). In these cases, it may be necessary to cancel and restart the process or pick a simpler selection.

## Neural Network Details
Below are the details concerning the structure of the neural network used in our files as well as the parameters found to be optimal for the general training process.
- **Structure**
  - Layout: Dense Neural Network 2-50-50-2
  - Activations: ReLU Hidden Layers, SoftMax Output Layer
  - Regularization: Max-Norm Weight Contraint with l=4 for Hidden Layers
  - Initialization: Parameters Chosen from Gausian Distr.
    - Biases: mean = 0, std. = 1
    - Weights: mean = 0, std. = 1/sqrt(n_in)
    
- **Training Parameters**
  - Cost Function: Cross Entropy
  - Optimization: Momentum Based Gradient Descent with Backpropogation
    - Momentum: friction_coeff = .5
    - Learning Rate: eta = .3
  - Training Perscription: trained in loop, each pass using 100 epochs over the entire dataset

## Files Included
- "Classification_Visualizer.py": Script for creating the visualization
- "network.py": File containinig classes for constructing simple, dense neural networks. Modified slightly from its original state found in "Digit_Classifier" Repository.
- "Images/": Folder containing images used for this README.
