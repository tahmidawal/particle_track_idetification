import matplotlib.pyplot as plt
import matplotlib as plt1
import numpy as np
import os
import click
from datetime import datetime
import sys
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PowerTransformer

@click.command()
@click.option('--npy_input_filename')
@click.option('--power_transform')
@click.option('--log_scale')
def data_preprocessing(npy_input_filename,power_transform,log_scale):
    """This file will take the all_events_all_particles_array_shuffled.npy
    file and do some of the final steps of data preprocessing on it. The end result will
    be 
        1) training input array
        2) validation/test input array
        3) training target array
        4) validation/test target array
    
    These numpy files will be stored to the same directory that 
    all_events_all_particles_array_shuffled.npy was stored in. 
    
    Parameters: 
        -npy_input_filename : string, which the user should input 
                as <all_events_all_particles_array_shuffled.npy>
        -power_transform: string 1 or 0. 1 if you want to power transform the data. 
                0 otherwise
        -log_scale: string 1 or 0. 1 if you want to log_scale the data. 0 if not. 
    
    Note that this script will not both power tranform and log scale the data. 
    
    This script will also min-max scale the resulting power-transformed or
    log-scaled data. 
    
    Example command line input:
    
    python data_preprocessing.py --npy_input_filename=all_events_all_particles_array_shuffled.npy --power_transform=0 --log_scale=1
    
    """
    
    #turn the input strings 1 and 0 to integers so that
    #the logical if elif statement below actually works: 
    log_scale = int(log_scale)
    power_transform = int(power_transform)
    
    #You see here that OUTPUT_DIR = INPUT_DIR
    #but feel free to change that. 
    OUTPUT_DIR = '../../Data/smallDataPreprocessingFiles'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    INPUT_DIR = '../../Data/smallDataPreprocessingFiles'
    
    #input file path will actually allow us to access the 
    #all events array:
    input_file_path = os.path.join(INPUT_DIR, npy_input_filename)
    #load the array, and allow_pickle because it is an object.
    all_events_data = np.load(input_file_path, allow_pickle = True)
    
    #here we seperate the features (namely, the 108 by 112 images)
    #from the 113th target column. These remained aligned even though
    #they are seperated, such that the mapping from 
    #image to targets is not altered:
    target_data_unsplit = all_events_data[:,0,112]
    input_data_unsplit = all_events_data[:,:,:112]

    click.echo("Padding the data with 4 total rows of zeros so the images are 112x112 instead of 108x112.")
    #pad the data with 2 rows above and 2 rows below per image. 
    #we do this padding because the CNN later on will require square images. 
    temp = np.zeros(112*len(input_data_unsplit)).reshape(len(input_data_unsplit),1,112)
    for i in range(2):
        input_data_unsplit = np.append(temp, input_data_unsplit, axis= 1)
    for i in range(2):
        input_data_unsplit = np.append(input_data_unsplit, temp, axis= 1)
        
    #we will have 20% test/validation data, and 80%
    #training data. 
    train_split_index = (len(input_data_unsplit) // 5) * 4
    
    #doing the actual splitting:
    train_input = input_data_unsplit[0:train_split_index]
    train_target = target_data_unsplit[0:train_split_index]
    val_input = input_data_unsplit[train_split_index:]
    val_target = target_data_unsplit[train_split_index:]
    
    
    #I decided it would be good to add a date/time stamp 
    #to each power tranform or log_scale saved file. 
    today = datetime.today()
    click.echo("starting feature scaling")
    
    if power_transform == log_scale:
        sys.exit("power_transform cannot be the same boolean value as log_scale")
    elif power_transform == 1:
        #here we want to power tranform with respect to 
        #a single feature, so we must collapse 
        #the entire train_input array into one that only has 
        #1 column. 
        train_input = train_input.reshape(len(train_input)*112*112,1)
        
        #we will use this power_transform instance for both the 
        #val_input set and the train_input set. This comes 
        #from the sklearn power transformer method. 
        #https://scikit-learn.org/stable/modules/generated/
        #sklearn.preprocessing.PowerTransformer.html
        pt = PowerTransformer(method='yeo-johnson')
        pt.fit(train_input)
        train_input= pt.transform(train_input)

        #do the same to the validation set
        val_input = val_input.reshape(len(val_input)*112*112,1)
        val_input= pt.transform(val_input)
        
        #now we reshape the power transformed arrays to there original
        #shapes. 
        train_input = train_input.reshape(len(train_input)//(112*112),112,112)
        val_input = val_input.reshape(len(val_input)//(112*112),112,112)
        
        #min max scale:
        maximum = np.amax(train_input)
        train_input = train_input*(1.0/(maximum/2.0)) - 1.0

        maximum = np.amax(val_input)
        val_input = val_input*(1.0/(maximum/2.0)) - 1.0
        
        #now we do some string manipulations to make the 
        #filenames for these power transformed input arrays
        #and save them to the intended folder
        t_input_filename = str(today) + "train_input_power_trsfm.npy"
        t_input_filename = os.path.join(OUTPUT_DIR, t_input_filename)
        v_input_filename = str(today) + "val_input_power_trsfm.npy"
        v_input_filename = os.path.join(OUTPUT_DIR, v_input_filename)
        np.save(t_input_filename, train_input)
        np.save(v_input_filename, val_input)
        
    elif log_scale == 1:
        click.echo("inside log_scale conditional.")
        
        #when log scaling, we cannot take the 
        #log of zero, which many of the pixels are, 
        #so we add 1 to each pixel, so that the log of 
        #1 being 0 matches up with how that pixel was 
        #already 0 before. Adding 1 to each pixel, since some
        #of the charges are in the thousands, doesn't do too much 
        #harm to the data. 
        train_input = np.log(train_input.astype(float)+1)
        val_input = np.log(val_input.astype(float)+1)
        
        #min max scale:
        maximum = np.amax(train_input)
        train_input = train_input*(1.0/(maximum/2.0)) - 1.0

        maximum = np.amax(val_input)
        val_input = val_input*(1.0/(maximum/2.0)) - 1.0
        
        #adding timestamp to the filenames and saving to the 
        #correct folder. 
        t_input_filename = str(today) + "train_input_log_scale.npy"
        t_input_filename = os.path.join(OUTPUT_DIR, t_input_filename)
        v_input_filename = str(today) + "val_input_log_scale.npy"
        v_input_filename = os.path.join(OUTPUT_DIR, v_input_filename)
        np.save(t_input_filename, train_input)
        np.save(v_input_filename, val_input)
        
   
    #here we save the targets as constant filenames because these don't really
    #change each time we run this script, so long as the input 
    #npy file stays the same. 
    train_targets_path = os.path.join(OUTPUT_DIR, "train_targets.npy")
    val_targets_path = os.path.join(OUTPUT_DIR, "val_targets.npy")
    np.save(train_targets_path, train_target)
    np.save(val_targets_path, val_target)
    
if __name__ == '__main__':
    data_preprocessing()