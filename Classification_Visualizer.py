# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:43:40 2020

@author: Mitchell

Classification_Visualizer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The purpose of this script is to build an interactive GUI for visualizing
classification using neural networks. This is done by allowing the user to
select a set of points from a figure of data using Matplotlib's LassoSelector.
We then iteratively use an ANN with structure [2,50,50,2] to classify the data
into two groups: 1) the selected points and 2) the unselected points. The 
process of this iterative classification is visualized in the second figure
by displaying the classification regions at each training step. At any point
the user has the option to stop the training by hitting backspace, at which
point they may select a new set of points and hit enter to restart training.
Simple shapes can be identified by the process in just a few training sessions.
More complex shapes, especially for disjoint sets and close points, may take a
few minutes to complete.

Network Structure and Trainining Parameters:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Network: 
   -Structure: Two hiddden layers of size 50 - (2,50,50,2)
   -Activations: ReLU hidden layers and SoftMax output layer
   -Initialization: Random starting weights/biases chosen from gaussina distr.
       -Biases: mean = 0, std = 1
       -Weights: mean = 0, std = 1/sqrt(n_in)

 Training:
   -Cross Entropy Cost Function
   -Gradient Descent with Momentum
       -Learning Speed: Eta = .3
       -Friction: .5
   -Trained in loop each using 100 epochs with each covering the entire dataset
   -Regularization:
       -Max-Norm weight constraint with l=4
"""
#imports
import network
import numpy as np
from matplotlib.widgets import LassoSelector
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# Class for selecting points using Matplotlib's Lasso Selector Widget
# See Lasso Selector tutorial for this code
class SelectFromCollection:
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))
        
        lineprops = {'color': 'red', 'lw': 2}
        self.lasso = LassoSelector(ax, onselect=self.onselect, lineprops=lineprops)
        self.ind = []

    def onselect(self, verts):
        global count, patch, reset
        if count !=0:
            patch.remove()
        path = Path(verts)
        reset = 1
        patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor = 'red')
        count+=1
        ax1.add_patch(patch)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    
    #ignoring divide by 0 erros which are common but not an issue
    np.seterr(divide='ignore', invalid='ignore')
    
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # Datapoint to choose from for cluster
    data = np.random.rand(100, 2)

    # Initializing our interactive figure/visualizater
    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=subplot_kw, figsize=(13, 6))
    pts1 = ax1.scatter(data[:, 0], data[:, 1], s=80)
    ax1.set_title("Select Points Using Cursor, Press Enter to Accept")
    ax2.set_title("ANN Classification: Backspace to Cancel")
    fig.subplots_adjust(bottom = .2)
    text1 = fig.text(0.10,0.05, 'Accuracy:', fontsize=20)
    text2 = fig.text(0.40,0.05, 'Cost:', fontsize=20)
    text3 = fig.text(0.70,0.05, 'Status:', fontsize=20)
    
    # Setting some global variables to help with conditions for certain actions
    # and updating the figures
    count=0
    patch = []
    reset = 0
    escaper = False
    
    # Initiate our LassoSelector
    selector = SelectFromCollection(ax1, pts1)
    
    # Command for executing our neural network for classifying the data points
    # does the bulk of the work in the script
    def accept(event):
        global selector, escaper, text1, text2, text3
        # conditions for accepted commands (enter to run and backspace to cancel)
        if event.key == "enter" and reset == 1:
            fig.canvas.draw()
            escaper = False
            
            # Build our network and training set using points
            net = network.Network([2,50,50,2])
            training_data = []
            for i in range(selector.Npts):
                if i in selector.ind:
                    training_data.append((np.array(selector.xys[i]).reshape(2,1),
                                         np.array([[0.],[1.]])))
                else:
                    training_data.append((np.array(selector.xys[i]).reshape(2,1),
                                         np.array([[1.],[0.]])))
                    
            # Iteratively run Stochastic Gradient Decent to train our network
            # Runs until all the points are succesfully classified or the
            # user cancels the training session
            results = []
            counter = 0
            while results != list(selector.ind) and escaper==False:
                # Clear our figure and replot points
                ax2.clear()
                pts2 = ax2.scatter(data[:, 0], data[:, 1], s=80)
                ax2.set_title("ANN Classification: Backspace to Cancel")  
                fc = pts2.get_facecolors()
                fc = np.tile(fc, (selector.Npts, 1))
                counter += 1
                # Stochastic Gradient Descent (see network for details)
                net.SGD(training_data, 100, 
                        100, .3, max_norm = 4,
                        friction = .5)
                # Updating accuracy and status on figure
                accuracy = net.accuracy(training_data,convert=True)
                cost = net.total_cost(training_data, lmbda=0.)
                if cost > 2:
                    net = network.Network([2, 50, 50, 2])
                    counter = 0
                text1.remove()
                text2.remove()
                text3.remove()
                text1 = fig.text(0.10,0.05, 'Accuracy: {}%'.format(accuracy), fontsize=20)
                text2 = fig.text(0.40,0.05, 'Cost: %.3f'%cost+'%', fontsize=20)
                if accuracy == 100:
                    text3 = fig.text(0.70,0.05, 'Status: Finished', fontsize=20)
                else: 
                    text3 = fig.text(0.70,0.05, 'Status: Running', fontsize=20)
                # Using countourf plot to visualize the classifcation of our
                # enitre dataset with the neural network
                results1 = np.array(net.evaluate(training_data))
                results = [i for i in range(len(results1)) if results1[i]==1]
                fc[:, -1] = .3
                fc[np.where(results1==1.), -1] = 1
                pts2.set_facecolors(fc)
                N = 100
                x0 , xend = ax1.get_xlim()
                y0 , yend = ax1.get_ylim()
                dx , dy = (xend-x0)/N , (yend-y0)/N
                X,Y = np.meshgrid(np.linspace(x0,xend,N+1),np.linspace(y0,yend,N+1))
                contour_set = []
                for  x, y in zip(X,Y):
                    contour_data = []
                    for i in range(N+1):
                        contour_data.append((np.array([[x[i]],[y[i]]]),
                                              np.array([[1.0],[0.]])))
                    results_c = net.evaluate(contour_data)
                    contour_set.append(results_c)
                Z = np.array(contour_set)
                levels = [0, 0.5, 1]
                cf = ax2.contourf(X + dx/2.,
                      Y + dy/2., Z, levels = levels, 
                      colors=((1.,0.,0.),(0.,1.,0.)),
                      alpha = .3)
                # Updats the figures and pauses script for 1 second
                plt.pause(1.)
            
            # For when training session is canceled
            if escaper == True:
                text3.remove()
                text3 = fig.text(0.70,0.05, 'Status: Canceled', fontsize=20)
                plt.pause(.1)
        # Command for canceling traininig session
        if event.key == "backspace":
            escaper = True
            
    # Allows our first figure to capture mouse click and tie it to Lasso
    fig.canvas.mpl_connect("mouse_press_event", accept)
    # Allows us to capture key presses for starting and canceling training
    fig.canvas.mpl_connect("key_press_event", accept)

    # Renders figure
    plt.show()