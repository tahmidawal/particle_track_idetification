import click 
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def numpy_array_to_3d_arrays():
    """
    This function will load in the numpy arrays created by txt_to_numpy_array_3D.py and turn the numpy arrays into data that can be used as input for neural networks. The data will be placed in a folder containing the data.
    The numpy arrays are converted into 2 different types of data: data for general use and data for plotting
    These new numpy arrays will be placed in two new folders
    
    Inputs: None
    Returns: None
    """
    
    #we will predefine pathfiles for the different data sets, feel free to change folder names
    directory_general_3D_data = '../../Data/3DArrays/' #this data will be the baseline data, it can be used for further processing if Michelle or Raghu assign you to different neural network model architetures
    #makes the output folder in case it doesn't exist already:
    if not os.path.exists(directory_general_3D_data):
        os.makedirs(directory_general_3D_data)
    
    directory_plotting_data_arrays = '../../Data/3DplottingArrays/' #plotting will be mostly for jupyter notebook use to visualize data
    #makes the output folder in case it doesn't exist already: 
    if not os.path.exists(directory_plotting_data_arrays):
        os.makedirs(directory_plotting_data_arrays)
    
    
    #load the data
    #you might need to change the folder directory if you changed folder names or filenames
    click.echo("loading data")
    deuteron = np.load('../../Data/3DsmallDataParticleChargeArrays/deuteron_array_3D.npy', allow_pickle=True)
    he3 = np.load('../../Data/3DsmallDataParticleChargeArrays/he3_array_3D.npy', allow_pickle=True)
    he4 = np.load('../../Data/3DsmallDataParticleChargeArrays/he4_array_3D.npy', allow_pickle=True)
    proton = np.load('../../Data/3DsmallDataParticleChargeArrays/proton_array_3D.npy', allow_pickle=True)
    triton = np.load('../../Data/3DsmallDataParticleChargeArrays/triton_array_3D.npy', allow_pickle=True)
    click.echo("data loaded")
    
    #we then seperate the time and charge data from the 1500 events of each particle into time and charge arrays
    click.echo("seperating time and charge data")
    #deuteron
    time_deuteron = []
    charge_deuteron = []
    for event in range(len(deuteron)):
        time_deuteron.append(deuteron[event][0])
        charge_deuteron.append(deuteron[event][1])

    #he3    
    time_he3 = []
    charge_he3 = []
    for event in range(len(he3)):
        time_he3.append(he3[event][0])
        charge_he3.append(he3[event][1])

    #he4    
    time_he4 = []
    charge_he4 = []
    for event in range(len(he4)):
        time_he4.append(he4[event][0])
        charge_he4.append(he4[event][1])

    #proton    
    time_proton = []
    charge_proton = []
    for event in range(len(proton)):
        time_proton.append(proton[event][0])
        charge_proton.append(proton[event][1])

    #triton    
    time_triton = []
    charge_triton = []
    for event in range(len(triton)):
        time_triton.append(triton[event][0])
        charge_triton.append(triton[event][1])
    click.echo("time and charge data seperated")
    
    #from the jupyter file TestingDataSet.ipynb, the dimensions of a single event is as follows:
    #rows: 108, columns: 112, time: uncapped float value
    #we will define some dimensions as constants:
    num_events = len(triton) #should be 1500
    num_rows = len(triton[0][0]) #should be 108
    num_cols = len(triton[0][0][0]) #should be 112
    
    
    #basically, for each particle, there are 1500 events. in the 1500 events, there is a 2D detector pad represented by a 2 dimensional matrix with either charge or time data. the charge and time data is seperated, but they have the exact same dimensions, as each curly bracket, {}, represents an individual detector in the 2D detector. the third dimension comes in as time, which can be interpreted as a third dimension because the drift velocity of electrons in the gas chamber is constant, so there is no acceleration, and thus we can treat time as a accurate representation of a z-coordinate.
    #we determined that the highest value of time was 290.84 in the jupyter notebook.
    #we floor divide time by 3, because a 108 by 112 by 291 system would be disproportionate. floor divisor subject to change later
    #290.84//3 = 96 <-- we use 97 as the upper bound of the z-coordinate 
    
    #we create different types of data to use. the arrays simply titled like deuteron_3d are for general use, while the arrays titled like only_detected_events_deuteron are used for plotting for data visualization. feel free to comment out the only_detected_events arrays if it takes too long to run, but it really shouldnt. it is just nice to have different formats of data for different uses, as different neural networks may want different input formats
    #the code is not optimized, and was mostly written this way for the previous author, Ian, for better visualization. feel free to optimize any of the code in this file if future ALPhAers want to do so :) 
    
    #all data is in these arrays, even empty {} 
    deuteron_3d = [] 
    he3_3d = []
    he4_3d = []
    proton_3d = []
    triton_3d = []
    
    #only data where there was a particle detection
    only_detected_events_deuteron = [] 
    only_detected_events_he3 = []
    only_detected_events_he4 = []
    only_detected_events_proton = []
    only_detected_events_triton = []
    
    click.echo("creating numpy arrays for folders 3DArrays and 3DplottingArrays")
    for event in tqdm(range(num_events)):
        single_event_deuteron = []
        single_event_he3 = []
        single_event_he4 = []
        single_event_proton = []
        single_event_triton = []

        single_event_only_detected_events_deuteron = []
        single_event_only_detected_events_he3 = []
        single_event_only_detected_events_he4 = []
        single_event_only_detected_events_proton = []
        single_event_only_detected_events_triton = []

        for row in range(num_rows):
            for col in range(num_cols):
                if time_deuteron[event][row][col] != []:
                    single_event_deuteron.append([row, col, int(time_deuteron[event][row][col][0]//3), charge_deuteron[event][row][col][0]])
                    single_event_only_detected_events_deuteron.append([row, col, int(time_deuteron[event][row][col][0]//3), charge_deuteron[event][row][col][0]])
                else:
                    single_event_deuteron.append([row, col, 0, 0])
                if time_he3[event][row][col] != []:
                    single_event_he3.append([row, col, int(time_he3[event][row][col][0]//3), charge_he3[event][row][col][0]])
                    single_event_only_detected_events_he3.append([row, col, int(time_he3[event][row][col][0]//3), charge_he3[event][row][col][0]])
                else:
                    single_event_he3.append([row, col, 0, 0])
                if time_he4[event][row][col] != []:
                    single_event_he4.append([row, col, int(time_he4[event][row][col][0]//3), charge_he4[event][row][col][0]])
                    single_event_only_detected_events_he4.append([row, col, int(time_he4[event][row][col][0]//3), charge_he4[event][row][col][0]])
                else:
                    single_event_he4.append([row, col, 0, 0])
                if time_proton[event][row][col] != []:
                    single_event_proton.append([row, col, int(time_proton[event][row][col][0]//3), charge_proton[event][row][col][0]])
                    single_event_only_detected_events_proton.append([row, col, int(time_proton[event][row][col][0]//3), charge_proton[event][row][col][0]])
                else:
                    single_event_proton.append([row, col, 0, 0])
                if time_triton[event][row][col] != []:
                    single_event_triton.append([row, col, int(time_triton[event][row][col][0]//3), charge_triton[event][row][col][0]])
                    single_event_only_detected_events_triton.append([row, col, int(time_triton[event][row][col][0]//3), charge_triton[event][row][col][0]])
                else:
                    single_event_triton.append([row, col, 0, 0])
        deuteron_3d.append(single_event_proton)
        he3_3d.append(single_event_he3)
        he4_3d.append(single_event_he4)
        proton_3d.append(single_event_proton)
        triton_3d.append(single_event_triton)

        only_detected_events_deuteron.append(single_event_only_detected_events_deuteron)
        only_detected_events_he3.append(single_event_only_detected_events_he3)
        only_detected_events_he4.append(single_event_only_detected_events_he4)
        only_detected_events_proton.append(single_event_only_detected_events_proton)
        only_detected_events_triton.append(single_event_only_detected_events_triton)
    click.echo("arrays created")
    #running this line of code might cause kernal to die in jupyter, so take caution. it may be fine in python script tho
    #we turn the lists into numpy arrays, so they can be saved into a folder
    #we turn everything into floats here because numpy arrays are all supposed to contain the same type, so be aware that if you wish to treat them as integers, you will need to convert those specific values in the arrays in seperate code
    click.echo("converting arrays and saving them to numpy arrays with floats")
    deuteron_3d_floats = np.array(deuteron_3d, dtype = float)
    he3_3d_floats = np.array(he3_3d, dtype = float)
    he4_3d_floats = np.array(he4_3d, dtype = float)
    proton_3d_floats = np.array(proton_3d, dtype = float)
    triton_3d_floats = np.array(triton_3d, dtype = float)
    
    only_detected_events_deuteron_floats = np.array(only_detected_events_deuteron)
    only_detected_events_he3_floats = np.array(only_detected_events_he3)
    only_detected_events_he4_floats = np.array(only_detected_events_he4)
    only_detected_events_proton_floats = np.array(only_detected_events_proton)
    only_detected_events_triton_floats = np.array(only_detected_events_triton)
    
    #only_detected_events_deuteron_floats = np.array(only_detected_events_deuteron, dtype = float)
    #only_detected_events_he3_floats = np.array(only_detected_events_he3, dtype = float)
    #only_detected_events_he4_floats = np.array(only_detected_events_he4, dtype = float)
    #only_detected_events_proton_floats = np.array(only_detected_events_proton, dtype = float)
    #only_detected_events_triton_floats = np.array(only_detected_events_triton, dtype = float)
    
    #we define the file name and combine the file name and directory to make a file path
    file_name_deuteron_3d_floats = 'deuteron_3d_floats.npy'
    file_path_deuteron_3d_floats = os.path.join(directory_general_3D_data, file_name_deuteron_3d_floats)
    file_name_he3_3d_floats = 'he3_3d_floats.npy'
    file_path_he3_3d_floats = os.path.join(directory_general_3D_data, file_name_he3_3d_floats)
    file_name_he4_3d_floats = 'he4_3d_floats.npy'
    file_path_he4_3d_floats = os.path.join(directory_general_3D_data, file_name_he4_3d_floats)
    file_name_proton_3d_floats = 'proton_3d_floats.npy'
    file_path_proton_3d_floats = os.path.join(directory_general_3D_data, file_name_proton_3d_floats)
    file_name_triton_3d_floats = 'triton_3d_floats.npy'
    file_path_triton_3d_floats = os.path.join(directory_general_3D_data, file_name_triton_3d_floats)

    file_name_only_detected_events_deuteron_floats = 'only_detected_events_deuteron_floats.npy'
    file_path_only_detected_events_deuteron_floats = os.path.join(directory_plotting_data_arrays, file_name_only_detected_events_deuteron_floats)
    file_name_only_detected_events_he3_floats = 'only_detected_events_he3_floats.npy'
    file_path_only_detected_events_he3_floats = os.path.join(directory_plotting_data_arrays, file_name_only_detected_events_he3_floats)
    file_name_only_detected_events_he4_floats = 'only_detected_events_he4_floats.npy'
    file_path_only_detected_events_he4_floats = os.path.join(directory_plotting_data_arrays, file_name_only_detected_events_he4_floats)
    file_name_only_detected_events_proton_floats = 'only_detected_events_proton_floats.npy'
    file_path_only_detected_events_proton_floats = os.path.join(directory_plotting_data_arrays, file_name_only_detected_events_proton_floats)
    file_name_only_detected_events_triton_floats = 'only_detected_events_triton_floats.npy'
    file_path_only_detected_events_triton_floats = os.path.join(directory_plotting_data_arrays, file_name_only_detected_events_triton_floats)
    
    #we save the files to the directories defined earlier above
    np.save(file_path_deuteron_3d_floats, deuteron_3d_floats)
    np.save(file_path_he3_3d_floats, he3_3d_floats)
    np.save(file_path_he4_3d_floats, he4_3d_floats)
    np.save(file_path_proton_3d_floats, proton_3d_floats)
    np.save(file_path_triton_3d_floats, triton_3d_floats)
    
    np.save(file_path_only_detected_events_deuteron_floats, only_detected_events_deuteron_floats)
    np.save(file_path_only_detected_events_he3_floats, only_detected_events_he3_floats)
    np.save(file_path_only_detected_events_he4_floats, only_detected_events_he4_floats)
    np.save(file_path_only_detected_events_proton_floats, only_detected_events_proton_floats)
    np.save(file_path_only_detected_events_triton_floats, only_detected_events_triton_floats)
    
    click.echo("arrays converted and saved")

    
    #end of code
    
    
def main():
    numpy_array_to_3d_arrays()

if __name__ == "__main__":
    main()