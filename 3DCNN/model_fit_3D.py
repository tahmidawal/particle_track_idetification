import matplotlib.pyplot as plt
import numpy as np
import os

# there is an error with tensorflow if we dont add lines 7 to 9, as memory usage of the script is very high
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import click 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, MaxPool3D , Flatten, Dropout
import sparse as sp
from datetime import datetime


class toDense(tf.keras.layers.Layer):
    """
    Converts an input batch, which is a sparse tensor, into a dense layer for the model to read in.
    
    Class: 
    toDense KERAS LAYER
        Custom class to be added to the model for batch processing before training
    
    Inputs: 
    inputs (SPARSE TENSOR):
        Batch of sparse tensor from input data
    
    Returns: 
    tf.sparse.to_dense(input) (DENSE):
        A dense array for input into the 3D CNN model
    """
    def call(self, input):
        if isinstance(input,  tf.sparse.SparseTensor):
            return tf.sparse.to_dense(input)
        return input


def model_build(batch_size):
    """
    Constructs a basic 3DCNN with based on the VGG16 architecture. 
    
    Inputs:
    batch_shape (INT):
    
    
    Returns: 
    model (KERAS MODEL):
        3DCNNModel based on VGG16
    """
    
    #6000 is the number of training events
    input_shape = (112,112,112,1)
    batch_shape = (batch_size,112,112,112,1)
   
    #5 is the number of classes, 
    num_classes = 5
    
    #change the number of filters and layers you want here, but keep in mind more filters means longer run time and more parameters, while less filters means less accuracy and less parameters
    
    #some of the layers are commented out, because I wanted to see how removing them would affect overfitting and accuracy
    DIVISOR = 32
    
    model = Sequential()
    model.add(tf.keras.Input(batch_shape=batch_shape))
    model.add(toDense())
    model.add(Conv3D(input_shape=input_shape,filters=64/DIVISOR,kernel_size=(3,3,3),padding="same", activation="relu"))
    model.add(Conv3D(filters=64/DIVISOR,kernel_size=(3,3,3),padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    model.add(Conv3D(filters=128/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(Conv3D(filters=128/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Dropout(0.5))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    model.add(Conv3D(filters=256/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(Conv3D(filters=256/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(Conv3D(filters=256/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Dropout(0.5))
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Dropout(0.5))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(units = 256/DIVISOR, activation="relu"))
    model.add(Dense(units = 256/DIVISOR, activation="relu"))
    #model.add(Dense(units = 512/DIVISOR, activation="relu"))
    #model.add(Dense(units= 512/DIVISOR, activation="relu"))
    model.add(Dense(units= num_classes, activation="softmax"))
    
    return model
    
def load_data():
    """
    Loads in data from foler 3DinputCNNArrays
    
    Inputs: None
    
    
    Returns: 
    train_inputs (COO ARRAY):
        training inputs for model input
     
    train_labels (NUMPY ARRAY):
        training labels for model input
    """
    #load in the data from the folder 3DinputCNNArrays
    click.echo("loading data")
    train_inputs = sp.load_npz('../../Data/3DinputCNNArrays/train_inputs.npz')

    #save arguments needed to convert to sparse tensor
    indices = list(zip(*train_inputs.coords))
    values = train_inputs.data
    dense_shape = train_inputs.shape
    
    #turn COO sparse array to sparse tensor
    train_inputs = tf.sparse.SparseTensor(indices,values,dense_shape)
    
    train_labels = np.load('../../Data/3DinputCNNArrays/train_labels.npy', allow_pickle=True)
    click.echo("data loaded")
    
    return train_inputs, train_labels
    
    
def fit_model(model, train_inputs, train_labels, today_time, epochs, lr, batch_size):
    """
    Fits the model, also saves the model and produces a graph for loss curves in the same folder
    
    Inputs: 
    model (KERAS MODEL):
        model used for training on inputs and labels
       
    train_inputs (COO ARRAY):
        training inputs for model input
     
    train_labels (NUMPY ARRAY):
        training labels for model input   
       
    today_time (STRING):
        string of a time value for matching directories
        
    epochs (INT):
        number of epochs to train for
        
    lr (FLOAT):
        learning rate of the model
        
    batch_size (INT):
        batch size for fitting
        
    Returns: 
    results.history['loss']: (FLOAT ARRAY)
        array containing training loss
    """
    #compile the model
    model.compile(tf.keras.optimizers.Adam(lr=lr), 
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
    
    #define directory for the weights trained in the model to be saved in
    DIRECTORY = '../../ModelCheckpoints/3DmodelCheckpoints/results{}'.format(today_time)

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    # The checkpoints will be saved with the corresponding epoch number in their filename
    ckpt_path = os.path.join(DIRECTORY, 'weights_epoch_{epoch:02d}.ckpt')

    # Setup checkpoint callback. We save the weights.
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path)
    my_callbacks = [ckpt_callback]
    
    click.echo("training model")
    results = model.fit(train_inputs, train_labels, batch_size = batch_size, epochs = epochs, callbacks=my_callbacks)
    
    return results.history['loss']

def plotting_and_validation(model, loss, today_time, epochs, batch_size):
    """
    Calculates Validation Loss and creates graph with training and validation loss together 
    
    Inputs: 
    model (tf.keras.model):
        3DCNNModel based on VGG16
        
    loss (tf.keras.callbacks.History):
        for plotting training loss
        
    today_time (STRING):
        string of a time value for matching directories
        
    epochs (INT):
        number of training epochs
    
    batch_size (INT):
        batch size for validation
    Returns: None
    """
    
    #location of the checkpoints, and where we want to save the loss curve in
    DIRECTORY = '../../ModelCheckpoints/3DmodelCheckpoints/results{}'.format(today_time)
    
    #load in the data from the folder 3DinputCNNArrays
    click.echo("loading validation data")
    validation_inputs = sp.load_npz('../../Data/3DinputCNNArrays/validation_inputs.npz')
    
    #save arguments needed to convert to sparse tensor
    indices = list(zip(*validation_inputs.coords))
    values = validation_inputs.data
    dense_shape = validation_inputs.shape
    
    #turn COO sparse array to sparse tensor
    validation_inputs = tf.sparse.SparseTensor(indices,values,dense_shape)
    
    validation_labels = np.load('../../Data/3DinputCNNArrays/validation_labels.npy', allow_pickle=True)
    click.echo("data loaded")
    
    #create array for validation loss
    val_loss_array = np.zeros(epochs, dtype=float)
    
    #since we cannot get validation loss from the model.fit function, we will have to manually create the validation loss curve by taking each of the checkpoints, evaluating the loss on validation inputs, and then making those predictions, and then calculate the validation loss in a for loop, which will put the validation losses for each epoch into an array for plotting with the normal training loss
    for epoch in range(1,epochs + 1):
    #for single digits, the epoch number is like 01, 02 etc. So we have to attach a zero to each integer
        if epoch < 10:
            epoch_string = '0' + str(epoch)
        else:
            epoch_string = str(epoch)
        
        #define directory where we are taking the weights from
        epoch_DIRECTORY = DIRECTORY + '/weights_epoch_{}.ckpt'.format(epoch_string)
        
        #load our checkpoints
        checkpoint = tf.train.Checkpoint(model)
        click.echo("restoring checkpoint")
        
        #the ".expect_partial()" makes sure that not all the variables are loaded, because some variables will be unchanged
        checkpoint.restore(epoch_DIRECTORY).expect_partial()
        
        click.echo("epoch:" + epoch_string)
        click.echo("calculating loss")
        predictions = np.array(model.predict(validation_inputs, batch_size = batch_size, verbose=1), dtype = float)
        #turn the labels into floats, if it is not, the code will fail
        np.array(validation_labels, dtype = float)
        
        #we use the loss function CategoricalCrossentropy to evaluate the validation loss
        val_loss = tf.keras.losses.CategoricalCrossentropy()
        val_loss = val_loss(validation_labels, predictions).numpy()
        print('Current Val Loss:', val_loss)
        #adding the validation loss to the array used for plotting, val_loss
        val_loss_array[epoch - 1] = val_loss
        val_loss = None

    #plotting
    file_name = "traininglosscurve.png" #feel free to change the file name
    save_fig_path = os.path.join(DIRECTORY, file_name)
    
    #plots the training loss and validation loss together in one graph
    plt.plot(loss, label="training loss")
    plt.plot(val_loss_array, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_fig_path, format='png')

def main():
    #choose the batch size here. higher batch size means faster model, but will take up more memory. expect higher batch sizes to run out of memory
    batch_size = 64
    #feel free to change the starting learning rate
    lr = 1E-5
    #feel free to change no. of epochs for how long you will train your model
    epochs = 1000 
    
    #this is for formatting purposes, as the folder with checkpoint data has the timestamps of when it was created
    today_time = str(datetime.today())
    
    #build the model
    model = model_build(batch_size)
    
    #PLEASE MAKE SURE YOU AREN'T LOADING A CHECKPOINT IF YOU DONT WANT TO
    #if you want to load in checkpoints from a previous model, change the load_model boolean here to be true, otherwise keep it at false
    load_model = False
    
    if load_model == True:
        #enter folder you want to take checkpoints for, you will have to manually do this depending on what folder you want to use
        folder = 'results2022-05-12 02:20:25.611168'
        #enter the epoch of the checkpoint you want to restore here
        desired_epoch = 1000
        DIRECTORY = '../../ModelCheckpoints/3DmodelCheckpoints/' + folder + '/weights_epoch_{}.ckpt'.format(desired_epoch)
        click.echo("restoring previous model checkpoint")
        checkpoint = tf.train.Checkpoint(model)
        #this makes sure that not all the variables are loaded, because some variables will be unchanged
        checkpoint.restore(DIRECTORY).expect_partial()
 
    #load in the data
    train_inputs, train_labels= load_data()
    #fit model
    
    
    loss = fit_model(model, train_inputs, train_labels, today_time, epochs, lr, batch_size)
    plotting_and_validation(model, loss, today_time, epochs, batch_size)
    click.echo("end of code")
    
if __name__ == "__main__":
    main()
