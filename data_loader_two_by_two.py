import numpy as np

def get_data_sets():

    examples = [np.array([[1, 1],
                          [1, 1]]),
                np.array([[1, 0],
                          [1, 0]]),
                np.array([[1, 1],
                          [0, 0]]),
                np.array([[0, 1],
                          [0, 1]]),
                np.array([[0, 0],
                          [0, 0]]),
                np.array([[0, 1],
                          [1, 0]]),
                np.array([[1, 0],
                          [1, 1]]),
                np.array([[0, 0],
                          [0, 1]]) ]

    def training_set():
    
        index_to_choose = np.random.randint(len(examples))     
        yield examples[index_to_choose]           
 
    def evaluation_set():

        index_to_choose = np.random.randint(len(examples))     
        yield examples[index_to_choose]           

    return training_set, evaluation_set



