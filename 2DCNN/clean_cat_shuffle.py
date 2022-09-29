import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plt1
import os
import click
from tqdm import tqdm

@click.command()
@click.option('--he3')
@click.option('--he4')
@click.option('--p')
@click.option('--t')
@click.option('--d')
def clean_cat_shuffle(he3,he4,p,t,d):
    """
    
    Just make sure to use the arrays from the summed arrays folder.
    This file takes in five npy files, each storing a summed array
    and gets rid of each image/event that is all zeros. 
    There are actually a lot of these for some reason. 
    You will see a reduction in size of the dataset for a single array
    from about 
    1500 to about 1300, and likely in this proportion for differing sized 
    datasets. 
    
    Then this file concatenates a 113th column to each image, which
    stores a number between 0 and 4 depending on which particle the array represents. 
    0 for proton, 1 for deteuron, 2 for triton, 3 for He3, 4 for He4. 
    
    Then the file concatenates these five particle arrays together
    and shuffles them up. 
    
    It then saves these into a new folder with the filename 
    "all_events_all_particles_array_shuffled.npy"
    
    Parameters: 
        -he3: .npy file storing the summed array from the sum_pads_arrays.npy file. 
        -he4: .npy file ^
        -p: .npy file^
        -t: .npy file^
        -d: .npy file^
    
    example command line input: 
    python clean_cat_shuffle.py --he3=he3.npy  --p=proton.npy 
    --t=triton.npy --d=deuteron.npy --he4=he4.npy"""
    
    #output directory where the output file will be stored:
    OUTPUT_DIR = "../../Data/smallDataPreprocessingFiles"
    
    #make the folder if it doesn't already exist:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    #where to get the arrays from: 
    #IMPORTANT TO CHANGE THIS TO THE CORRECT FOLDER NAME:
    #though it should work if no changes were made orginally
    SUMMED_DIR = "../../Data/smallDataSummedChargeArrays"
    
    #making file paths:
    he3_path = os.path.join(SUMMED_DIR, he3)
    he4_path = os.path.join(SUMMED_DIR, he4)
    p_path = os.path.join(SUMMED_DIR, p)
    t_path = os.path.join(SUMMED_DIR, t)
    d_path = os.path.join(SUMMED_DIR, d)
    
    click.echo("Starting the process of loading particle arrays.")
    #load up the arrays: 
    d_array = np.load(d_path, allow_pickle=True)
    he3_array = np.load(he3_path, allow_pickle=True)
    he4_array = np.load(he4_path, allow_pickle=True)
    p_array = np.load(p_path, allow_pickle=True)
    t_array = np.load(t_path, allow_pickle=True)
    

    #Proton: 0, Deuteron: 1, Triton: 2, He3: 3, He4: 4
    #order of my_list matters because it determines the target index of each particle. 
    my_list = [p_array, d_array, t_array, he3_array, he4_array]
    click.echo("Starting the process of removing all-zeros images and adding a target column.")
    index = 0
    #iterate through the five arrays:
    for i in tqdm(my_list):
        
        sum_of_event = []
        
        #get the sum of each event/image
        for j in i:
            sum_of_event.append(np.sum(j))
        
        #eliminate the images of sum being zero (means all pixels were 0.)
        sum_array_1d = np.array(sum_of_event)
        i = i[sum_array_1d.nonzero()]
        
        #get the shape of the array so we can add
        #the target column:
        i_axis0, i_axis1, i_axis2 = i.shape
        #here, temp gets formed into an array of shape (# of events, rows per image, 1)
        #this is precisely the shape of the 113th column that we want
        temp = np.ones(i_axis1*i_axis0).reshape(i_axis0,i_axis1,1)
        
        #multiplying by the index counter turns the 113th column of 
        #ones into an integer from 0-4, which will identify that specific image. 
        temp = temp*index
        
        #append on our column to the original array:
        i = np.append(i, temp, axis= 2)
        
        #error checking: 
        shape_i = i.shape
        shape_i = str(shape_i)
        toPrint = "expecting 113 columns:" + shape_i
        click.echo(toPrint)
        
        #update the array outside of for loop instance
        my_list[index] = i
        
        #update index: 
        index+=1
        
        

    click.echo("Starting the process of concatenating particle arrays.")
    #here we concatenate all the arrays together. Since only the 
    #arrays in the list were altered (thus have 113 columns on third axis), 
    #we have to specify 
    #my_list[x] instead of d_array,..,ect...
    all_events_data = np.concatenate((my_list[0],my_list[1],my_list[2],my_list[3],my_list[4]))
    
    #shuffle 
    np.random.shuffle(all_events_data)
    
    #error checking to see that there are 113 columns in all events array:
    shape_all = all_events_data.shape
    shape_all = str(shape_all)
    toPrint = "expecting 113 columns:" + shape_all
    click.echo(toPrint)
    
    
    #free up space in memory: 
    d_array = None
    t_array = None
    p_array = None
    he4_array = None
    he3_array = None

    #save the new all events shuffled array to a new folder:
    file_path = os.path.join(OUTPUT_DIR, "all_events_all_particles_array_shuffled.npy")
    np.save(file_path, all_events_data)



if __name__ == '__main__':
    clean_cat_shuffle()