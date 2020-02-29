import numpy as np

class Dense(object):

    def __init__(self, num_inputs=None, num_outputs=None):
        self.num_inputs = num_inputs + 1 # We add one because we always have an extra node in every layer which is a bias neuron which will be always activated
        self.num_outputs = num_outputs
        # It is important to be visualizing what is happening in this dense layer. Think of a layer as the one row of neurons aligned in a vertical line. The inputs are the number of neurons in the previous layer, aka the number of connections which will be coming in to every neuron. The outputs number is going to be how many neurons are in this layer. So we will need a specification for every single output node of the weight with which to multiply and add together the signals from every input layer of the previous layer! That is what our weight matrix is specifying!
        self.weight_matrix = np.random.sample([self.num_inputs, self.num_outputs])
        # Just initializing the actual neuron values to 0, they will get updated
        self.input_values = np.zeros(self.num_inputs)
        self.output_values = np.zeros(self.num_outputs)
