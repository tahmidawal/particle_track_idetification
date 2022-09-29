import click 
import os
import numpy as np
import sklearn
from sklearn.utils import shuffle
from tqdm import tqdm
import sparse as sp

def numpy_array_to_3d_cnn():
    """
    This function will load in the numpy arrays created by npy_array_to_3Ddata.py and turn the numpy arrays into data that can be used as input for a 3D convulational neural network. The new 3DCNN array and labels that correspond to the 3DCNN array will be saved in thhe folder 3DinputCNNArrays. Both arrays will be randomized in unison.
    
    Inputs: None
    Returns: None
    """

    #we will predefine pathfiles for the different data sets, feel free to change folder names
    directory_input_arrays = '../../Data/3DinputCNNArrays' #randomized input arrays for a 3D CNN will be placed in this folder
    #makes the output folder in case it doesn't exist already: 
    if not os.path.exists(directory_input_arrays):
        os.makedirs(directory_input_arrays)

    click.echo("loading data")
    deuteron_3d_floats = np.load('../../Data/3DArrays/deuteron_3d_floats.npy', allow_pickle=True)
    he3_3d_floats = np.load('../../Data/3DArrays/he3_3d_floats.npy', allow_pickle=True)
    he4_3d_floats = np.load('../../Data/3DArrays/he4_3d_floats.npy', allow_pickle=True)
    proton_3d_floats = np.load('../../Data/3DArrays/proton_3d_floats.npy', allow_pickle=True)
    triton_3d_floats = np.load('../../Data/3DArrays/triton_3d_floats.npy', allow_pickle=True)
    click.echo("data loaded")
    
    #we will define some dimensions as constants:
    num_events = len(deuteron_3d_floats) #should be 1500
    num_rows = 108 #should be 108 #from npy_array_to_3Dcnndata.py, we get these integers for rows and cols
    num_cols = 112 #should be 112
    cube = 112 #we want the dimensions to all be the same so we can make a cube, this is because to turn the data into a sparse array, we will need to have data of the same dimensions
    
    #we will define a time bound, so the np.zeros function will create a three dimensional array
    time_bound = 97 #we choose 97 to be the bound because of what we said earlier in lines 88-90
    
    #remember from the lines above, we had to change all values in the arrays to floats to be consistent for numpy arrays, but we keep charge as a float
    click.echo("converting numpy arrays into 3D CNN input format")
    inputs = np.zeros((5*num_events, cube, cube, cube, 1), dtype = float) 
    
    for event in tqdm(range(num_events)):
        for mini_array in deuteron_3d_floats[event]:
            x_pos = int(mini_array[0])
            y_pos = int(mini_array[1])
            time = int(mini_array[2])
            charge = mini_array[3]
            if (charge != 0):
                inputs[event][x_pos][y_pos][time][0] = charge

        for mini_array in he3_3d_floats[event]:
            x_pos = int(mini_array[0])
            y_pos = int(mini_array[1])
            time = int(mini_array[2])
            charge = mini_array[3]
            if (charge != 0):
                inputs[event+1500][x_pos][y_pos][time][0] = charge

        for mini_array in he4_3d_floats[event]:
            x_pos = int(mini_array[0])
            y_pos = int(mini_array[1])
            time = int(mini_array[2])
            charge = mini_array[3]
            if (charge != 0):
                inputs[event+1500*2][x_pos][y_pos][time][0] = charge

        for mini_array in proton_3d_floats[event]:
            x_pos = int(mini_array[0])
            y_pos = int(mini_array[1])
            time = int(mini_array[2])
            charge = mini_array[3]
            if (charge != 0):
                inputs[event+1500*3][x_pos][y_pos][time][0] = charge
    
        for mini_array in triton_3d_floats[event]:
            x_pos = int(mini_array[0])
            y_pos = int(mini_array[1])
            time = int(mini_array[2])
            charge = mini_array[3]
            if (charge != 0):
                inputs[event+1500*4][x_pos][y_pos][time][0] = charge
    
    click.echo("data converted for 3D CNN use")
    
    #turning numpy array into a sparse array, we do so to save disk space and memory usage on data proccessing in the future
    click.echo("turning numpy array into a sparse array")
    inputs_sparse = sp.as_coo(inputs)
    click.echo("data converted into a sparse array")
    #free up memory
    inputs = None
    
    #we will define a dictionary for the particle types. it wont be actually used in this code, but for PointNet models, you might need to use it. it is also nice to have just for clarity on what index from 0-4 represents what particle 
    #class_map = {0: 'Deuteron', 1: 'Helium-3', 2: 'Helium-4', 3: 'Proton', 4: 'Triton'}

    #define the array for labels.
    
    click.echo("attaching labels to arrays and saving them into folder 3DinputCNNArrays")
    #these arrays will contain all the data from all five particle types
    labels = np.zeros((7500,5), dtype = float)
   
    #the labels will be in array format, as this is the format required for labels in a CNN
    #the order is different this time, because it is ordered in order of number of nucleons, which will look better on the confusion matrices
    proton = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) 
    deuteron = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    triton = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) 
    he_3 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    he_4 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    
    #this for loop goes through the events and labels and appends them to the corresponding array
    num_labels = num_events*5 #should be 7500
    for event in tqdm(range(num_labels)):
        if event < num_events + 1:
            labels[event] = deuteron
        elif event < num_events*2 + 1:
            labels[event] = he_3
        elif event < num_events*3 + 1:
            labels[event] = he_4
        elif event < num_events*4 + 1:
            labels[event] = proton
        elif event < num_events*5 + 1:
            labels[event] = triton
    click.echo("labels attached")
    
    #free up memory
    deuteron_3d_floats = None
    he3_3d_floats = None
    he4_3d_floats = None
    proton_3d_floats = None
    triton_3d_floats = None
    
    #we now randomize the data in unison #!!!I have commented out the code since randomizeing it here creates memory problems
    #click.echo("randomizing data")
    #inputs, labels = shuffle(inputs, labels)
    #click.echo("data randomized")
    
    #we want to save these files in 3DinputCNNArrays, so we name the array and join the directory to make a file path
    file_name_inputs = 'inputs.npz'
    file_name_labels = 'labels.npy'
    file_path_inputs = os.path.join(directory_input_arrays, file_name_inputs)
    file_path_labels = os.path.join(directory_input_arrays, file_name_labels)
    
    click.echo("saving data")
    sp.save_npz(file_path_inputs, inputs_sparse)
    np.save(file_path_labels, labels)
    click.echo("data saved")
    click.echo("job complete")
    #end of code
    
def main():
    numpy_array_to_3d_cnn()

if __name__ == "__main__":
    main()