import numpy as np
import matplotlib.pyplot as plt
from random import randint

""" Within this module, we will house the blueprints used to create our top level NN objects, an ANN"""

class ANN(object):
    def __init__(self, layers=None, actual_pixel_range=None):
        "Upon initiation of this ANN object, we must be supplied with a model which is a list of actual layers, such as a dense layer taken from layers.py"
        self.layers = layers  # list of actual layer object
        self.n_training_datum_to_generate = 10000
        self.n_eval_datum_to_generate = 10000 # number of training/eval data to generate
        self.actual_pixel_range = actual_pixel_range 
        self.target_pixel_range = {'low':-0.5, 'high':0.5} 
       
    def train(self, training_set):
        """The train method which is called once on the ANN object. This will repeat a training
        process a number of time, self.n_training_datum_to_generate. This process will get a
        piece of data, .ravel() it, normalize it, then run the forward propogate function on.
        See below for info on .forward_propogate()""" 
        self.iteration_errors_train = np.array([]) 
        for number in range(1, self.n_training_datum_to_generate+1):
            current_training_datum = self.normalize_datum(next(training_set()).ravel()) # .ravel() unravels my np.array into a 1D version!
            output_vector = self.forward_propogate(current_training_datum)
            self.iteration_errors_train = np.append(self.iteration_errors_train, randint(-5, 5)) # Stand-in for an error measurement
            if number % 1000 == 0:
                self.update_error_report(train_or_eval='train', bin_num=number/1000, save_path='output_figs/binned_errors_train.png')

    def evaluate(self, eval_set):
        "Same as above but for eval data instead of training data"
        self.iteration_errors_eval = np.array([]) 
        for number in range(1, self.n_eval_datum_to_generate+1):
            current_eval_datum = self.normalize_datum(next(eval_set()).ravel())
            output_vector = self.forward_propogate(current_eval_datum)
            self.iteration_errors_eval = np.append(self.iteration_errors_eval, randint(-5, 5)) 
            if number % 1000 == 0 and number>0:
                self.update_error_report(train_or_eval='eval', bin_num=number/1000, save_path='output_figs/binned_errors_eval.png')

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

    def forward_propogate(self, input_vector):
        """Forward propogate is some of the meat and potatoes of our NNs. This will first get
        an input datum vector, then it will .ravel() it, then it will turn it in to a matrix
        by inserting an axis along the first dimenions, so it's [1, some_number]"""
        output_vector = input_vector.ravel() # this input vector is likely just a numpy array, which is different than the matrix we want. The np.shape probably shows (,5) or something instead of (1,5). The (1, 5) really is a one dimensional matrix and that is important for us to have!
        output_vector = output_vector[np.newaxis, :] # inserting an axis along first dimension
        # Within the ANN_object.train() we call this function, which will take the input vector and send it through all the layers in a forward manner
        for layer_index in range(len(self.layers)):
            output_vector = self.layers[layer_index].forward_propogate(output_vector) 
        return output_vector.ravel()

    def update_error_report(self, train_or_eval=None, bin_num=None, save_path=None):
        """Finds the average error in bins of bin_size and generates a line plot showing this n_bin_ave_error over time
        """
        if train_or_eval=='train':
            binned_iteration_errors = np.hsplit(self.iteration_errors_train, bin_num)
            ave_errors = [np.average(iter_error_bin) for iter_error_bin in binned_iteration_errors]
            graph = plt.plot(range(len(ave_errors)), ave_errors) 
            plt.savefig(save_path) 
            plt.cla()
            plt.clf()
        else:
            binned_iteration_errors = np.hsplit(self.iteration_errors_eval, bin_num)
            ave_errors = [np.average(iter_error_bin) for iter_error_bin in binned_iteration_errors]
            graph = plt.plot(range(len(ave_errors)), ave_errors) 
            plt.savefig(save_path)
            plt.cla()
            plt.clf()
                    





