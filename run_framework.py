import numpy as np
import data_loader_two_by_two as dl
from nn_framework import framework 
from nn_framework import layer 
from nn_framework import activation
# This is the highest level. It is what imports from everywhere else, so you can start here and drill in to any of the packages or things called. At a high level, so far, it creates our training/eval data set generators, then gets the dimensions of a sample of our data, then sets the number of outputs, then creates an architecture which is just like a blueprint of the number of nodes per layer, then creates one dense layer using this architecture, then creates our list of the layer objects called layers. Then we create our high level object, the framework.ANN, and supply the previously list of layers. Then we can call this high level model's functions.

# Using the data loader module, load in some sample data.
# Remember these are generators you are saving in the objects training_set/testing_set
training_set, evaluation_set = dl.get_data_sets()

# Get the number of neurons long our input and output layers will be for every layer and create the architecture,
# which paints a picture of the number of layers in every layer of our network
input_nodes = len(np.reshape(next(training_set()), -1)) 
output_nodes = input_nodes

HL_nodes = [6, 3, 9]

architecture = [input_nodes]+HL_nodes+[output_nodes]
layers=[]
for layer_index in range(len(architecture)-1):
    layers.append(layer.Dense(num_inputs=architecture[layer_index], num_outputs=architecture[layer_index+1], activation_func=activation.tanh))
# Putting in the pixel range to help us normalize to a wanted range
actual_pixel_range = {'low':0,'high':1}

# Call our framework module from the nn_framework package and instantiate a high level ANN object 
autoencoder = framework.ANN(layers=layers, actual_pixel_range=actual_pixel_range)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
