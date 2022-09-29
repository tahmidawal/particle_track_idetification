import click 
import os
import numpy as np
from tqdm import tqdm

@click.command()
@click.option('--txtfilename')
@click.option('--npyfilename')
def txt_to_npy(txtfilename, npyfilename):
    """
    This file will convert the Monte Carlo generated charge and time data
    text files to numpy files which will be used further down the 
    data processing pipeline. Unless specified otherwise, it will put the 
    numpy array it creates in to a folder called 
    "./3DsmallDataParticleChargeArrays/". 
    
    Parameters:
        -txtfilename: the raw data txt file you want to turn into an array
        -npyfilename: the .npy filename you want the array to be stored to. 
    
    Example command line input:
    python txt_to_npy_array.py --txtfilename=output_p.txt --npyfilename=proton_array.npy
    
    
    Inputs: None
    Returns: None
    """
    
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
    
    click.echo("averaging time data")
    #take average of the time data
    for d in tqdm(range(len(data[0]))): #24000 iterations
        for j in range(len(data[0][0])): #108 iterations
            for k in range(len(data[0][0][0])):  #112 iterations
                data[0][d][j][k] = np.average(data[0][d][j][k])
    
    click.echo("summing charge data")
    #sum up the charges in the array
    #it is data[1] here because that is where the charge data is stored
    for d in tqdm(range(len(data[1]))): #24000 iterations
        for j in range(len(data[1][0])): #108 iterations
            for k in range(len(data[1][0][0])):  #112 iterations
                data[1][d][j][k] = np.sum(data[1][d][j][k])
    
    numpyArray = np.array(data, dtype=object)
    
    #make a file path with the out directory and the npyfilename from the command line:
    #file_path = os.path.join(PARTICLE_ARRAY_DIR, npyfilename)
    
    #save the array we created to the folder we made!
    np.save(npyfilename, numpyArray)
    
    click.echo("numpy array has been stored in ParticleChargeArrays folder.")
    

if __name__ == '__main__':
    txt_to_npy()