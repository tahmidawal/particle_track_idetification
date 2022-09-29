# Necessary imports for the code
import os
import numpy as np
import pylab as plt
import sklearn as skl
import pandas as pd
from random import sample
import click

#The number of points per event. In PointNet Data there has to be a fixed number of points N per event
# This is used as threshold to find number of points per event
N = 150 

BATCH_SIZE = 32 #  the size of the sub arrays in the training/test dataset that will contain the events

'''
This function loads data from the .npy files and and returns numpy arrays for further analysis. 
This data contains, the position, time and charge indentities in each events. 

Then numpy arrays containing the data for position, time, charge are processed for pointnet model. Only the coordinates of the data containing the trajectory of the particle are considered. Also, the charge data is not considered as we can't incorporate charge data with our current model.
'''
@click.command()
@click.argument('file-stem')
def dataprocessing(file_stem):
    
    # Path used to save the processed data
    directory_general_3D_data = file_stem + '/PointNetData1/'
    
    # If the directory doesn't already exit it will create directory 
    if not os.path.exists(directory_general_3D_data):
        os.makedirs(directory_general_3D_data) 
    
    '''
    Loads data from the .npy files 
    This data contains, the position, time and charge indentities in each events. 
    '''
    deuteron = np.load(file_stem + '/3DsmallDataParticleChargeArrays/deuteron_array_3D.npy', allow_pickle=True)
    he3 = np.load(file_stem + '/3DsmallDataParticleChargeArrays/he3_array_3D.npy', allow_pickle=True)
    he4 = np.load(file_stem + '/3DsmallDataParticleChargeArrays/he4_array_3D.npy', allow_pickle=True)
    proton = np.load(file_stem + '/3DsmallDataParticleChargeArrays/proton_array_3D.npy', allow_pickle=True)
    triton = np.load(file_stem + '/3DsmallDataParticleChargeArrays/triton_array_3D.npy', allow_pickle=True)
    

    # For each event for loops to append the z-dimensional data which is the time.
    time_deuteron = []
    for event in range(len(deuteron)):
        time_deuteron.append(deuteron[event][0])
   
    time_he3 = []
    for event in range(len(he3)):
        time_he3.append(he3[event][0])
  
    time_he4 = []
    for event in range(len(he4)):
        time_he4.append(he4[event][0])
  
    time_proton = []
    for event in range(len(proton)):
        time_proton.append(proton[event][0])

    time_triton = []
    for event in range(len(triton)):
        time_triton.append(triton[event][0])
        
   
    #only data where there was a particle detection will be stored in these arrays as PointNet model only needs the points where 
    #a particle will be present
    only_detected_events_deuteron = [] 
    only_detected_events_he3 = []
    only_detected_events_he4 = []
    only_detected_events_proton = []
    only_detected_events_triton = []
    
    num_events = len(triton) #should be 1500, this is used to create the for loop to append the point data points into the arrays
    num_rows = len(triton[0][0]) #should be 108
    num_cols = len(triton[0][0][0]) #should be 112
    # For each events only takes the points where the particle was present
    
    for event in range(num_events):

        single_event_only_detected_events_deuteron = []
        single_event_only_detected_events_he3 = []
        single_event_only_detected_events_he4 = []
        single_event_only_detected_events_proton = []
        single_event_only_detected_events_triton = []
        
        #we floor divide time by 3, because a 108 by 112 by 291 system would be disproportionate. floor divisor subject to change later
        #290.84//3 = 96 <-- we use 97 as the upper bound of the z-coordinate
        
        for row in range(num_rows):
            for col in range(num_cols):
                if time_deuteron[event][row][col] != []:
                    single_event_only_detected_events_deuteron.append([row, col, int(time_deuteron[event][row][col][0]//3)])
                    
                if time_he3[event][row][col] != []:
                    single_event_only_detected_events_he3.append([row, col, int(time_he3[event][row][col][0]//3) ])
                    
                if time_he4[event][row][col] != []:
                    single_event_only_detected_events_he4.append([row, col, int(time_he4[event][row][col][0]//3)])
                    
                if time_proton[event][row][col] != []:
                    single_event_only_detected_events_proton.append([row, col, int(time_proton[event][row][col][0]//3)])
                    
                if time_triton[event][row][col] != []:
                    single_event_only_detected_events_triton.append([row, col, int(time_triton[event][row][col][0]//3)])
              
    

       # There must only be a constant number of points in each of the events for the PointNet model to work successfully. Here, we randomly take N points from the events which have N or more than N points. 
        # The events containing less that N number of points are not used to train our model
        if len(single_event_only_detected_events_deuteron)>=N:
            only_detected_events_deuteron.append(sample(single_event_only_detected_events_deuteron,N))

        if len(single_event_only_detected_events_he3)>=N:
            only_detected_events_he3.append(sample(single_event_only_detected_events_he3,N))

        if len(single_event_only_detected_events_he4)>=N:
            only_detected_events_he4.append(sample(single_event_only_detected_events_he4,N))

        if len(single_event_only_detected_events_proton)>=N:
            only_detected_events_proton.append(sample(single_event_only_detected_events_proton,N))

        if len(single_event_only_detected_events_triton)>=N:
            only_detected_events_triton.append(sample(single_event_only_detected_events_triton,N))
        
    # We'll classify and label each event 
    # We use a for loop and label proton - 0, deuteron -1, triton - 2, he3 - 3, he4 - 4 while appending the event points to points array and the labels of the events to labels array below
    labels = [] 
    points = []

    for i in range(len(only_detected_events_proton)):
        points.append(only_detected_events_proton[i])
        labels.append(0)
    for i in range(len(only_detected_events_deuteron)):
        points.append(only_detected_events_deuteron[i])
        labels.append(1)
    for i in range(len(only_detected_events_triton)):
        points.append(only_detected_events_triton[i])
        labels.append(2)
    for i in range(len(only_detected_events_he3)):
        points.append(only_detected_events_he3[i])
        labels.append(3)
    for i in range(len(only_detected_events_he4)):
        points.append(only_detected_events_he4[i])
        labels.append(4)

    
    # Converting the arrays to numpy arrays 
    points_floats = np.array(points)
    labels_ints = np.array(labels)
    
    # Creates the file names and paths for saving the data
    file_name = 'points.npy'
    file_path = os.path.join(directory_general_3D_data, file_name)
    file_name_labels = 'labels.npy'
    file_path_labels = os.path.join(directory_general_3D_data, file_name_labels)
    
    #Saving the data
    np.save(file_path, points_floats)
    np.save(file_path_labels, labels_ints)


    
def main():
    dataprocessing()

if __name__ == "__main__":
    main()