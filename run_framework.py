import numpy as np
import data_loader_two_by_two as dl
from nn_framework import framework 
from nn_framework import layer 

# Using the data loader module, load in some sample data.
# Remember these are generators you are saving in the objects training_set/testing_set
training_set, evaluation_set = dl.get_data_sets()

# Get the number of neurons long our input and output layers will be and create the architecture,
# which paints a picture of the number of layers in every layer of our network
input_units = len(np.reshape(next(training_set()), -1)) 
output_units = input_units 
architecture = [input_units, output_units]
dense_layer = layer.Dense(num_inputs=architecture[0], num_outputs=architecture[1])
print(dense_layer.weight_matrix)
model = [dense_layer]
# Putting in the pixel range to help us normalize to a wanted range
actual_pixel_range = {'low':0,'high':1}

# Call our framework module from the nn_framework package and instantiate a high level ANN object 
autoencoder = framework.ANN(model=model, actual_pixel_range=actual_pixel_range)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
