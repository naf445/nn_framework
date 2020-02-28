import numpy as np

""" Within this module, we will house the blueprints used to create our top level NN objects, an ANN"""

class ANN(object):
    def __init__(self, model=None):
        self.layers = model
        self.n_training_datum_to_generate = 10
        self.n_eval_datum_to_generate = 10
        
    def train(self, training_set):
        for number in range(self.n_training_datum_to_generate):
            current_training_datum = next(training_set()).ravel() # .ravel() unravels my np.array into a 1D version!
            print(current_training_datum)

    def evaluate(self, eval_set):
        for number in range(self.n_eval_datum_to_generate):
            current_eval_datum = next(eval_set()).ravel() 
            print(current_eval_datum)
