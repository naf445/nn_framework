import numpy as np

""" Within this module, we will house the blueprints used to create our top level NN objects, an ANN"""

class ANN(object):
    def __init__(self, model=None, actual_pixel_range=None):
        self.layers = model
        self.n_training_datum_to_generate = 10
        self.n_eval_datum_to_generate = 10
        self.actual_pixel_range = actual_pixel_range 
        self.target_pixel_range = {'low':-0.5, 'high':0.5} 
       
    def train(self, training_set):
        for number in range(self.n_training_datum_to_generate):
            current_training_datum = self.normalize_datum(next(training_set()).ravel()) # .ravel() unravels my np.array into a 1D version!
            print(current_training_datum)
            print(self.un_normalize_datum(current_training_datum))

    def evaluate(self, eval_set):
        for number in range(self.n_eval_datum_to_generate):
            current_eval_datum = self.normalize_datum(next(eval_set()).ravel())
            print(current_eval_datum)

    def normalize_datum(self, un_normalized_datum):
        min_actual_value = self.actual_pixel_range["low"]
        max_actual_value = self.actual_pixel_range["high"]
        actual_range = max_actual_value - min_actual_value
        normalized_datum = (un_normalized_datum - min_actual_value) / actual_range
        target_min = self.target_pixel_range['low']
        target_max = self.target_pixel_range['high']
        target_range = target_max-target_min
        normalized_datum *= target_range
        normalized_datum += target_min
        return normalized_datum

    def un_normalize_datum(self, normalized_datum):
        target_min = self.target_pixel_range['low']
        target_max = self.target_pixel_range['high']
        target_range = target_max-target_min
        un_normalized_datum = (normalized_datum-target_min)*target_range
        min_actual_value = self.actual_pixel_range["low"]
        max_actual_value = self.actual_pixel_range["high"]
        actual_range = max_actual_value - min_actual_value
        un_normalized_datum*=actual_range
        un_normalized_datum+=min_actual_value
        return un_normalized_datum 




