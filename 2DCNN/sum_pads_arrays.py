import click 
import os
import numpy as np
from tqdm import tqdm

@click.command()
@click.option('--npyfilename')
@click.option('--output')
def sum_pads_arrays(npyfilename,output):
    """This file takes one of the numpy arrays created in 
    the txt_to_npy_array.py script and remember how each element of 
    the array was a tuple? Well that's because there were multiple
    charge hits on the corresponding pad in the detector, 
    so we will just want to know the total charge that hit a specific
    pad, which is why we sum the charge over each pad, thus turning 
    this tuple into a single numpy.float64 value. (don't quote me on
    the numpy.float64 part, although I remember this being the case.)
    We will then save this summed array into a new folder to be passed
    used in the next step of the pipeline. 
    
    Parameters: 
        -npyfilename: check the folder made in the txt_to_npy_array.py for 
                an array to input to this script. 
        -output: a filename ending in .npy. I recommend naming it <particle>.npy
    
    An example command line code for running this script would be:
    
    python sum_pads_arrays.py --npyfilename=deuteron_array.npy --output=deuteron.npy
    """
    
    #here is the output folder where we will put the summed array:
    PARTICLE_DIR = '../../Data/smallDataSummedChargeArrays/'
    
    #make the folder if it doesn't already exist. 
    if not os.path.exists(PARTICLE_DIR):
        os.makedirs(PARTICLE_DIR)

    #IMPORTANT TO GET THIS FOLDER NAME RIGHT:
    #this is the folder where the arrays created in the
    #txt_to_npy_array.py were stored:
    PARTICLE_ARRAY_DIR = '../../Data/smallDataParticleChargeArrays/'
    
    #create the file paths upfront: 
    file_path_input = os.path.join(PARTICLE_ARRAY_DIR, npyfilename)
    file_path_output = os.path.join(PARTICLE_DIR, output)
    
    click.echo("Starting to load numpy array")
    #load the numppy array. This can take a few seconds. 
    #note that allow_pickle = True because we are loading 
    #an object array
    array = np.load(file_path_input, allow_pickle=True)
    
    #this is for debugging: 
    #in the large data txt files (>1GB), there was 
    #a point at which we ran into an unsolvable error, 
    #possibly corrupted data, but the file would work if
    #we didnt try saving more than a certain amount. 
    #this is a non-problem for the small data files though. 
#     ENDPOINT = len(array)-100
#     array = array[0:ENDPOINT]
    
    click.echo("Starting to sum charges over each pad.")
    #sum charges over each pad. This takes substantial time. 
    #take about 2 minutes for 1500 examples. 30 minutes 
    #for 24000 examples. 
    for d in tqdm(range(len(array))): #24000 iterations
        for j in range(len(array[0])): #108 iterations
            for k in range(len(array[0][0])):  #112 iterations
                array[d][j][k] = np.sum(array[d][j][k])
    
    np.save(file_path_output, array)




if __name__ == '__main__':
    sum_pads_arrays()