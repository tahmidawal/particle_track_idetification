#imports
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import click
import sparse as sp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, MaxPool3D , Flatten

class toDense(tf.keras.layers.Layer):
    """
    Converts an input batch, which is a sparse tensor, into a dense layer for the model to read in.
    
    Class: 
    toDense KERAS LAYER:
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


def model_build():
    """
    Constructs a basic 3DCNN with based on the VGG16 architecture. 
    
    Inputs: None
    
    
    Returns: 
    model (KERAS MODEL):
        3DCNNModel based on VGG16
    """
    
    #6000 is the number of training events
    input_shape = (112,112,112,1)
    batch_shape = (64,112,112,112,1)
    #5 is the number of classes, 
    num_classes = 5
    
    DIVISOR = 32
    
    #ANY CHANGES YOU MAKE TO THE MODEL WILL HAVE TO BE THE SAME HERE
    #For example, if you changed the number of filters, you will have to change that here as well.
    #Layers have been commented out to match the model in model_fit_3D.py
    #You do not need to copy in dropout layers if they are present in model_fit_3D.py 
    model = Sequential()
    model.add(tf.keras.Input(batch_shape=batch_shape))
    model.add(toDense())
    model.add(Conv3D(input_shape=input_shape,filters=64/DIVISOR,kernel_size=(3,3,3),padding="same", activation="relu"))
    model.add(Conv3D(filters=64/DIVISOR,kernel_size=(3,3,3),padding="same", activation="relu"))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    model.add(Conv3D(filters=128/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(Conv3D(filters=128/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    model.add(Conv3D(filters=256/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(Conv3D(filters=256/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(Conv3D(filters=256/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
   #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(Conv3D(filters=512/DIVISOR, kernel_size=(3,3,3), padding="same", activation="relu"))
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=(2,2,2)))
    model.add(Flatten())
    model.add(Dense(units = 256/DIVISOR, activation="relu"))
    model.add(Dense(units= 256/DIVISOR, activation="relu"))
   # model.add(Dense(units = 512/DIVISOR, activation="relu"))
    #model.add(Dense(units= 512/DIVISOR, activation="relu"))
    model.add(Dense(units= num_classes, activation="softmax"))
    
    return model

def plot_confusion_matrix_augmented(y_true,
                          y_pred,
                          classes,
                          folder_directory,
                          title=None,
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    
    Adapted from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    Arguments:
        y_true: Real class labels.
        y_pred: Predicted class labels.
        classes: List of class names.
        folder_directory: where to plot is saved
        title: Title for the plot.
        cmap: Colormap to be used.
    
    Returns:
        None.
    """
    if not title:
        title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    file_name = "augmented_confusion_matrix.png"
    save_fig_path = os.path.join(folder_directory, file_name)
    plt.savefig(save_fig_path, format='png')
    
    
def plot_confusion_matrix_normalized(y_true,
                          y_pred,
                          classes,
                          folder_directory,
                          title=None,
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    
    Adapted from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    Arguments:
        y_true: Real class labels.
        y_pred: Predicted class labels.
        classes: List of class names.
        folder_directory: where to plot is saved
        title: Title for the plot.
        cmap: Colormap to be used.
    
    Returns:
        None.
    """
    if not title:
        title = 'Confusion matrix'
    
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')#'pred', 'true', 'all'
    

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    
    im = ax.imshow( cm.astype(float) , interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    file_name = "normalized_confusion_matrix.png"
    save_fig_path = os.path.join(folder_directory, file_name)
    plt.savefig(save_fig_path, format='png')
    
def load_data():
    """
    Loads in data from foler 3DinputCNNArrays
    
    Inputs: None
    
    
    Returns: 
    test_inputs (COO ARRAY):
        test inputs for model testing
     
    test_targets (NUMPY ARRAY):
        test labels for model testing
    """
    click.echo("loading data")
    #test inputs and targets
    test_inputs = sp.load_npz('../../Data/3DinputCNNArrays/test_inputs.npz')
    test_targets = np.load('../../Data/3DinputCNNArrays/test_labels.npy', allow_pickle = True)

    indices = list(zip(*test_inputs.coords))
    values = test_inputs.data
    dense_shape = test_inputs.shape

    #turn COO sparse array to sparse tensor
    test_inputs = tf.sparse.SparseTensor(indices,values,dense_shape)
    click.echo("data loaded")
    return test_inputs, test_targets

def model_testing(model, test_inputs,test_targets, folder_directory, max_epoch):
    """
    Loads in data from foler 3DinputCNNArrays
    
    Inputs:
    model (tf.keras.model):
        3D CNN model to load checkpoints into
        
    test_inputs (COO ARRAY):
        test inputs for model testing
     
    test_targets (NUMPY ARRAY):
        test labels for model testing
        
    folder_directory (STRING):
        string of the folder to save confusion matrices in
        
    max_epoch (INT):
        the epoch you would like to evaluate the confusion matrices for
    
    
    Returns: None
    """
    DIRECTORY = '../../ModelCheckpoints/3DmodelCheckpoints/' + folder_directory
    
    epoch_DIRECTORY = DIRECTORY + '/weights_epoch_{}.ckpt'.format(max_epoch)
        
    checkpoint = tf.train.Checkpoint(model)
    click.echo("restoring checkpoint")
    checkpoint.restore(epoch_DIRECTORY).expect_partial()
    
    click.echo("making predictions")
    predictions = model.predict(test_inputs, batch_size = 64, verbose=1)
    
    #Predictions is currently a probability distribution, so we take the index from that distribution with the highest probability with argmax
    predictions = np.argmax(predictions, axis=1) 
    test_targets = np.argmax(test_targets, axis=1)
    
    test_targets = test_targets.astype(np.float64)
    
    normalize='pred'
    norm_confusion = confusion_matrix(test_targets, predictions, normalize='true') #'pred', 'true', 'all'
    confusion = confusion_matrix(test_targets, predictions)
    print('Normalized:')
    print(norm_confusion)
    print()
    print('Augmented:')
    print(confusion)
    print()
    plot_confusion_matrix_normalized(test_targets, predictions, ["Protons", "Deuterons", "Triton","He3", "He4"],DIRECTORY)
    plot_confusion_matrix_augmented(test_targets, predictions, ["Protons", "Deuterons", "Triton","He3", "He4"],DIRECTORY)

def main():
    model = model_build()
    #please input the folder you would like to store the confusian matrices in
    #REMEMBER TO SAVE AFTER PUTTING IN THE FOLDER YOU WOULD LIKE TO USE
    folder_directory = 'results2022-05-12 02:20:25.611168'
    #enter the highest epoch or the epoch for which the checkpoints you would like to extract from
    max_epoch = 1000
    
    test_inputs, test_targets = load_data()
    
    model_testing(model, test_inputs, test_targets, folder_directory, max_epoch)
if __name__ == "__main__":
    main()