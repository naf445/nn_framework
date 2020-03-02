import numpy as np

class tanh(object):
    """
    """
    def __init__():
        pass

    def squash(input_array):
       """This is the evaluation of the tanh function, our squishing function.
       It is going to take advantage of some numpy things to take an array of inputs, and then
       apply the tanh squishing function to each of the inputs.
       np.exp(param1): calculates the exponential(e^param1) of all elements in the input array.
       """ 
       e_to_the_input = np.exp(input_array)
       e_to_the_neg_input = np.exp(-input_array) 
       output_array = (e_to_the_input - e_to_the_neg_input)/(e_to_the_input + e_to_the_neg_input)
       return output_array
