import numpy as np
import data_loader_two_by_two as dl
from nn_framework import framework 

# Using the data loader module, load in some sample data.
# Remember these are generators you are saving in the objects training_set/testing_set
training_set, evaluation_set = dl.get_data_sets()

# Get the number of neurons long our input and output layers will be and create the architecture,
# which paints a picture of the number of layers in every layer of our network
input_units = len(np.reshape(next(training_set()), -1)) 
output_units = input_units 
architecture = [input_units, output_units]

# Call our framework module from the nn_framework package and instantiate a high level ANN object 
autoencoder = framework.ANN(model=None)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
