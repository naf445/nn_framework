import numpy as np

class Dense(object):
    """This is a dense layer, we will prob have some other types of layers.
    This layer, upon initialization creates some self. methods, records the number of inputs and outputs we want for this layer, creates a weight matrix, and initializes the input and output activation values, currently 0."""
    def __init__(self, num_inputs=None, num_outputs=None, activation_func=None):
        self.num_inputs = num_inputs + 1 # We add one because we always have an extra node in every layer which is a bias neuron which will be always activated
        self.num_outputs = num_outputs
        # It is important to be visualizing what is happening in this dense layer. Think of a layer as the one row of neurons aligned in a vertical line. The inputs are the number of neurons in the previous layer, aka the number of connections which will be coming in to every neuron. The outputs number is going to be how many neurons are in this layer. So we will need a specification for every single output node of the weight with which to multiply and add together the signals from every input layer of the previous layer! That is what our weight matrix is specifying!
        self.weight_matrix = np.random.sample([self.num_inputs, self.num_outputs])
        # Just initializing the actual neuron values to 0, they will get updated
        self.input_values = np.zeros(self.num_inputs)
        self.output_values = np.zeros(self.num_outputs)
        self.activation_func = activation_func

    def forward_propogate(self, input_vector):
        """Given a one dimensional input vector, adds a bias term, then uses our np.newaxis trick
        to turn this in to a one dimensional matrix. Then we can matrix multiply by our weight
        matrix, and this is our output activations to send off to the next layer!"""
        input_vector = np.append(input_vector, 1) # Adding 1 as a bias term
        input_vector = input_vector[np.newaxis, :] # inserting an axis along first dimension
        self.output_vector = np.dot(input_vector, self.weight_matrix)
        self.output_vector = self.activation_func.squash(self.output_vector)
        return self.output_vector 

