# Necessary imports for the code
import os
import glob
import numpy as np
import pylab as plt
import sklearn as skl
import click
from tensorflow import keras
from tensorflow.keras import layers
from random import sample
from sklearn.model_selection import train_test_split


'''
This function takes the events and labels and randomly shuffles the data and returns .npy arrays containing
trainining events, training labels, test events and test labels. 
'''
@click.command()
@click.argument('file-stem')
def train_test_splitting(file_stem):
    
    # Path used to save the processed data
    directory_general_data = file_stem + '/TrainTestSplitData/'
    
    # If the directory doesn't already exit it will create directory
    if not os.path.exists(directory_general_data):
        os.makedirs(directory_general_data)
        
    
    #Loads data from the .npy files 
    #This data contains, the 3D positions data of the trajectory of the particle
    
    points = np.load(file_stem + '/PointNetData1/points.npy', allow_pickle=True)
    labels = np.load(file_stem + '/PointNetData1/labels.npy', allow_pickle=True)
    
    
    # Randomly shuffles the data and divides it for training and testing. Here test_size is 0.2 - 20% of the total data
    train_events_final, test_events, train_labels_final, test_labels = train_test_split(points, labels, test_size = 0.2, random_state=42)

    
    # Converting the arrays to numpy arrays 
    train_events_final_npy = np.array(train_events_final)
    test_events_npy = np.array(test_events)
    train_labels_final_npy = np.array(train_labels_final)
    test_labels_npy = np.array(test_labels)
    
    # Creates the file names and paths for savign the data
    file_name_train_events = 'train_events.npy'
    file_path_train_events = os.path.join(directory_general_data, file_name_train_events)
    file_name_train_labels = 'train_labels.npy'
    file_path_train_labels = os.path.join(directory_general_data, file_name_train_labels)
    file_name_test_events = 'test_events.npy'
    file_path_test_events = os.path.join(directory_general_data, file_name_test_events)
    file_name_test_labels = 'test_labels.npy'
    file_path_test_labels = os.path.join(directory_general_data, file_name_test_labels)
    
    #Saving the data
    np.save(file_path_train_events, train_events_final_npy)
    np.save(file_path_train_labels, train_labels_final_npy)
    np.save(file_path_test_events, test_events_npy)
    np.save(file_path_test_labels, test_labels_npy)
    
    
    
def main():
    train_test_splitting()
   
    
if __name__ == "__main__":
    main()

