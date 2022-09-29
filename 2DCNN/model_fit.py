import os
import numpy as np
import tensorflow as tf
import pylab as plt
import sklearn as skl
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split

def load_data():
    """
    Loads in data for input in a 2DCNN.
    
    Inputs:
    None
    
    Returns:
    train_input (NUMPY ARRAY)
        train inputs for VGG16
    
    train_target (NUMPY ARRAY)
        train target for VGG16
        
    num_classes (INT)
        number of classes
        
    input_shape (TUPLE)
        tuple representing the input shape of the train inputs
    """
    # You will need to change what name your input array is, since it changes depending on when the code was run 
    input_name = '/2022-03-22 19:19:00.099416train_input_log_scale.npy'
    
    train_input = np.load('../../Data/smallDataPreprocessingFiles' + input_name, allow_pickle=True)

    train_target = np.load('../../Data/smallDataPreprocessingFiles/train_targets.npy', allow_pickle = True)
    
    # Expands dimensions of data
    train_input = np.expand_dims(train_input, axis=3)
    train_input = np.repeat(train_input, 3, axis=3)
    
    # Determine the number of class labels
    num_classes = len(np.unique(train_target)) #should be 5
    input_shape = (112,112,3)
    
    return train_input, train_target, num_classes, input_shape

def build_pretrained_vgg_model(input_shape):
    """
    Constructs a CNN with a VGG16's convolutional base and two fully-connected hidden layers on top.
    The convolutional base is frozen (the weights can't be updated) and has weights from training on
    the ImageNet dataset.
    
    Inputs:
    input_shape (TUPLE)
        tuple representing the input shape of the train inputs
    
    Returns:
    model (tf.keras.Model)
        VGG 16 model
    """
    # This loads the VGG16 model from TensorFlow with ImageNet weights
    vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # First we flatten out the features from the VGG16 model
    net = tf.keras.layers.Flatten()(vgg_model.output)

    # We create a new fully-connected layer that takes the flattened features as its input
    net = tf.keras.layers.Dense(512, activation=tf.nn.relu)(net)
    # And we add one more hidden layer
    net = tf.keras.layers.Dense(512, activation=tf.nn.relu)(net)
    
    # CAN REMOVE
    #Adding one more layer to see if there is any change in f1 scores
    net = tf.keras.layers.Dense(512, activation=tf.nn.relu)(net)

    # Then we add a final layer which is connected to the previous layer and
    # groups our images into one of the three classes, num_classes is 5 here
    output = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(net)

    # Finally, we create a new model whose input is that of the VGG16 model and whose output
    # is the final new layer we just created
    model = tf.keras.Model(inputs=vgg_model.input, outputs=output)
    
    # We loop through all layers except the last four and specify that we do not want 
    # their weights to be updated during training. Again, the weights of the convolutional
    # layers have already been trained for general-purpose feature extraction, and we only
    # want to update the fully-connected layers that we just added.
    for layer in model.layers[:-4]:
        layer.trainable = True
        
    return model

def fit_model(model, train_input, train_target, today_time):
    """
    Compiles the model and restructures data with one hot encoding, adds callbacks and learning rate reduction, and then trains the data. While the code does not return anything, it creates loss curves and saves the model into the same folder for later use when constructing confusion matrices from the model history. 

    Inputs: 
    model (tf.keras.model):
        2D CNN model
        
    train_input (NUMPY ARRAY)
        train inputs for VGG16
    
    train_target (NUMPY ARRAY)
        train target for VGG16
        
    today_time (STRING):
        string of a time value for matching directories
    
    Returns: None
    
    """
    model.compile(tf.keras.optimizers.Adam(lr=10e-4), 
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
    
    # This is the directory where model weights will be saved. Feel free to change it.
    DIRECTORY = '../../ModelCheckpoints/2DmodelCheckpoints/results-'
    DIRECTORY = DIRECTORY + today_time

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    # The checkpoints will be saved with the corresponding epoch number in their filename
    ckpt_path = os.path.join(DIRECTORY, 'weights.epoch.{epoch:02d}')

    # Setup checkpoint callback. We only save the weights, not the entire model
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True)
    
    #here's a call back that reduces the learning rate when the model's accuracy starts to plateau. 
    #change the minimum learning rate if you want to
    learning_rate = 10e-14
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau("val_accuracy", 
                                                     factor=0.5, patience=3, min_lr = learning_rate, mode="min") 
    
    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5),ckpt_callback,reduce_lr]
    
    
    enc = OneHotEncoder()

    one_hot_encoding_train_target = train_target.reshape(len(train_input),1)
    print(one_hot_encoding_train_target.shape)
    X = [[0], [1], [2], [3], [4]]
    # X = [['Proton', 0], ['Deuteron', 1], ['Triton', 2], ['He3', 3], ['He4', 4]]

    enc.fit(X)
    enc.categories_
    one_hot_encoding_train_target = enc.transform(one_hot_encoding_train_target).toarray()
    
    zero = np.array([1.0,0.0,0.0,0.0,0.0])
    one = np.array([0.0,1.0,0.0,0.0,0.0])
    two = np.array([0.0,0.0,1.0,0.0,0.0])
    three = np.array([0.0,0.0,0.0,1.0,0.0])
    four = np.array([0.0,0.0,0.0,0.0,1.0])
    my_dict = {0.0:zero, 1.0:one, 2.0: two, 3.0:three, 4.0: four}
    index = 0
    count = 0
    for i in one_hot_encoding_train_target:
        if (i[0] != my_dict[train_target[index]][0] or 
                           i[1] != my_dict[train_target[index]][1] or
                           i[2] != my_dict[train_target[index]][2] or
                           i[3] != my_dict[train_target[index]][3] or
                           i[4] != my_dict[train_target[index]][4]):
            count += 1
        index += 1
    
    
    results = model.fit(train_input,
          one_hot_encoding_train_target,
          batch_size=32,
          epochs=100,
          validation_split=0.2,
          callbacks=my_callbacks);

    model.save(DIRECTORY)
    
    plot_learning_curve(results.history)
    
    
    
def plot_learning_curve(history, today_time):
    """
    Plots learning curve for a 2D CNN model
    
    Inputs: 
    history (tf.keras.callbacks.History):
        results for plotting training and validation loss
        
    today_time (STRING):
        string of a time value for matching directories
    
    Returns: None
    """
    LEARNING_CURVE_DIR = '../../ModelCheckpoints/2DmodelCheckpoints/results-' + today_time

    if not os.path.exists(LEARNING_CURVE_DIR):
        os.makedirs(LEARNING_CURVE_DIR)

    #want to save figure with the data and time.
    file_name = "learningcurve{}{}".format(today_time,'.png')
    #file_name = "power_transform_LC{}".format(today_time,'.png')
    LEARNING_CURVE_DIR = LEARNING_CURVE_DIR + today_time
    save_fig_path = os.path.join(LEARNING_CURVE_DIR, file_name)
    
    plt.plot(history["loss"], label="training loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_fig_path, format='png')
    
    
    
def main():
    today_time = str(datetime.today())
    train_input, train_target, num_classes, input_shape = load_data()
    model = build_pretrained_vgg_model(input_shape, today_time)
    fit_model(model, train_input, train_target, today_time)
   
    
if __name__ == "__main__":
    main()
