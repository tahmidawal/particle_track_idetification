import click 
import os
import numpy as np
from tqdm import tqdm

@click.command()
@click.option('--npyfilename')
@click.option('--txtfilename')
def txt_to_npy(txtfilename, npyfilename):
    """This file will convert the Monte Carlo generated charge and time data
    text files to numpy files which will be used further down the 
    data processing pipeline. Unless specified otherwise, it will put the 
    numpy array it creates in to a folder called 
    "./smallDataParticleChargeArrays/". 
    
    Parameters:
        -txtfilename: the raw data txt file you want to turn into an array
        -npyfilename: the .npy filename you want the array to be stored to. 
    
    Example command line input:
    python txt_to_npy_array.py --txtfilename=output_p.txt --npyfilename=proton_array.npy"""
    
    PARTICLE_ARRAY_DIR = '../../Data/smallDataParticleChargeArrays/'
    
    #makes the output folder in case it doesn't exist already: 
    if not os.path.exists(PARTICLE_ARRAY_DIR):
        os.makedirs(PARTICLE_ARRAY_DIR)
    
    #opening and grasping the contents of the txt file:
    with open(txtfilename,"r") as f:
        contents = f.read()
    click.echo("file has been read, will now begin creating data list.")
    data = []
    level = -1
    number = ''
    
    #tqdm is a progress bar which can be seen in the terminal. It is very 
    #useful for knowing how long this code will take. This for loop
    #processes the text file making a quadruply nested list with a tuple
    #as each element. For instance, data[0][0][0][0] is a tuple. 
    #the second axis is of length 2. the first row is for time data, while
    #the second if for charge. We harvest all of it here, but we will 
    #slice out time data below, because we are not advanced enough 
    #to consider the 3rd dimension yet...
    for i in tqdm(contents):
        #print(i)
        if i == 'N':
            data.append([])
        elif i == '{':
            if level == -1:
                data[-1].append([])
                level -= 1
            elif level == -2:
                data[-1][-1].append([])
                level -= 1
            elif level == -3:
                data[-1][-1][-1].append([])
                level -= 1
            elif level == -4:
                print('Error 2')
        elif i == '}':
            level += 1
        elif i == ',':
            if level == -4:
                data[-1][-1][-1][-1].append(float(number))
            else:
                print('Error 3')
                print(level)
            number = ''
        elif i == '.' or i.isdigit():
            number += i
            
        
    click.echo("data list has been created, will begin transferring to numpy array.")
    #numpyArray = np.array(data,dtype=list)
    
    #store the list as an array
    #specifying dtype = object is important for allow this 
    #pipeline to work for the small data files
    numpyArray = np.array(data, dtype=object)
    
    #slice out the time data: 
    numpyArray = numpyArray[:,1]
    click.echo("numpy array has been created and time dimension has been eliminated.")
    
    #make a file path with the out directory and the npyfilename from the command line:
    file_path = os.path.join(PARTICLE_ARRAY_DIR, npyfilename)
    
    #save the array we created to the folder we made!
    np.save(file_path, numpyArray)
    click.echo("numpy array has been stored in ParticleChargeArrays folder.")
    

if __name__ == '__main__':
    txt_to_npy()