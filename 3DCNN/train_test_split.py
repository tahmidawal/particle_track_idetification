import click 
import os
import numpy as np
from sklearn.model_selection import train_test_split
import sparse as sp

def traintestsplit():
    """
    This function will load in the numpy arrays created by npy_array_to_3DCNNdata.py and perform a train-test split of 0.2. In addition, the train data will be split again for training and validation.
    
    Inputs: None
    Returns: None
    """

    #we will predefine pathfiles for the different data sets, feel free to change folder names
    directory_input_arrays = '../../Data/3DinputCNNArrays/'
    
    #makes the output folder in case it doesn't exist already: 
    if not os.path.exists(directory_input_arrays):
        os.makedirs(directory_input_arrays)
    
    #we load in the data created by npy_array_to_3Dcnndata.py
    click.echo("loading data")
    inputs = sp.load_npz('../../Data/3DinputCNNArrays/inputs.npz')
    labels = np.load('../../Data/3DinputCNNArrays/labels.npy', allow_pickle=True)
    click.echo("data loaded")
    
    #some functions have a hard time processing such large files, so we will make a train test split of the indices of the data, then re-use those for later
    #there are a total of 7500 results, so we set an array of [0,1,2,3,4....,7498,7499]
    indices = np.linspace(0,7499, num = 7500, dtype= int)
    
    #we perform a train-test split of 0.2
    click.echo("doing train test split")
    train_indices, test_indices = train_test_split(indices, test_size = 0.2)
    click.echo("train test split done")

    #we set the new arrays to be ordered based on the indices split
    train_inputs = inputs[train_indices]
    test_inputs = inputs[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    #we further split the training data into training and validation
    indices = np.linspace(0,5999, num = 6000, dtype= int)
    train_indices, validation_indices = train_test_split(indices, test_size = 0.2)
    
    train_inputs_new = train_inputs[train_indices]
    validation_inputs = train_inputs[validation_indices]
    train_labels_new = train_labels[train_indices]
    validation_labels = train_labels[validation_indices]
    
    
    #free up memory
    inputs = None
    labels = None
    
    #we want to save these files in 3DinputCNNArrays, so we name the array and join the directory to make a file path
    file_name_train_inputs = 'train_inputs.npz'
    file_name_validation_inputs = 'validation_inputs.npz'
    file_name_test_inputs = 'test_inputs.npz'
    file_name_train_targets = 'train_labels.npy'
    file_name_validation_targets = 'validation_labels.npy'
    file_name_test_targets = 'test_labels.npy'
    
    file_path_train_inputs = os.path.join(directory_input_arrays, file_name_train_inputs)
    file_path_validation_inputs = os.path.join(directory_input_arrays, file_name_validation_inputs)
    file_path_test_inputs = os.path.join(directory_input_arrays, file_name_test_inputs)
    file_path_train_targets = os.path.join(directory_input_arrays, file_name_train_targets)
    file_path_validation_targets = os.path.join(directory_input_arrays, file_name_validation_targets)
    file_path_test_targets = os.path.join(directory_input_arrays, file_name_test_targets)
    
    #and we save the data into an folder. keep in mind that the data has not be shuffled yet, as that is to be done in the jupyter notebook
    click.echo("saving data")
    sp.save_npz(file_path_train_inputs, train_inputs_new)
    sp.save_npz(file_path_validation_inputs, validation_inputs)
    sp.save_npz(file_path_test_inputs, test_inputs)
    np.save(file_path_train_targets, train_labels_new)
    np.save(file_path_validation_targets, validation_labels)
    np.save(file_path_test_targets, test_labels)
    click.echo("3D CNN arrays with labels saved in 3DinputCNNArrays")
    click.echo("job complete")
    #end of code
    
def main():
    traintestsplit()

if __name__ == "__main__":
    main()