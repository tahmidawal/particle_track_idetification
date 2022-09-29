# Necessary imports for the code
import os
import glob
import numpy as np
import tensorflow as tf
import pylab as plt
import sklearn as skl
import pandas as pd
import click
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

'''
Used to shuffle points within each events
'''
def augment(points, label):
    points = tf.random.shuffle(points)
    return points, label

'''
The functions below including 

1. conv_bn(x, filters)
2. dense_bn(x, filters)
3. OrthogonalRegularizer(keras.regularizers.Regularizer)
4. tnet(inputs, num_features)
5. build_point_cloud_model(NUM_POINTS, NUM_CLASSES)

are used to build our PointNet Neural Network. To learn more or understand this model better please visit either or both 
of these resourses.
https://keras.io/examples/vision/pointnet/
https://www.youtube.com/watch?v=GGxpqfTvE8c&t=878s
'''
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def build_point_cloud_model(NUM_POINTS, NUM_CLASSES):
    
    inputs = keras.Input(shape=(NUM_POINTS, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

'''
This function utilizes the train_dataset to train the model and 
uses test_dataset to plot the learning curve and confusion matrix
'''
def fit_model(model, train_dataset, test_dataset, file_stem):
    
    classes = ['Proton',  'Deuteron' ,'Triton',  'Helium-3',  'Helium-4'] # The classes according to their labels respectively

    # Compiles the model with the given learning rate.
    model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
    metrics=["sparse_categorical_accuracy"],
    )
    
    #Defined the directory
    DIRECTORY = file_stem + '/CheckPoints'
    
    #Creates the directory if there is none
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    # The checkpoints will be saved with the corresponding epoch number in their filename
    ckpt_path = os.path.join(DIRECTORY, 'weights.epoch.{epoch:02d}')

    # Setup checkpoint callback. We only save the weights, not the entire model
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True)
    
    learning_rate = 1e-15
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau("val_accuracy", 
                                                     factor=0.5, patience=3, min_lr = learning_rate, mode="min") 
    
    my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5),
    ckpt_callback,
    reduce_lr]
    
    #Uses train data to train the model
    result = model.fit(train_dataset, epochs=100, validation_data=test_dataset) 
    plot_learning_curve(result.history, file_stem) # Plots the learning curve as it trains the model throughout the epochs
    
    
    #Converts the test dataset to numpy arrays for use in confusion matrix function
    test_labels = []
    pred_labels = []
    
    data = test_dataset.take(len(test_dataset)) # Storing data from test dataset
    
    for i in range(len(test_dataset)):
        
        points, labels = list(data)[i]
        labels = labels.numpy()

        # run test data through model
        preds = model.predict(points)
        preds = tf.math.argmax(preds, -1)
        preds = preds.numpy()
        test_labels = test_labels+labels.tolist()
        pred_labels = pred_labels+ preds.tolist()
    
    test_labels = np.array(test_labels)
    pred_labels = np.array(pred_labels)
    
    # Plots confusion matrix for the result on test data
    plot_confusion_matrix(file_stem, test_labels, pred_labels, classes, cmap=plt.cm.Blues)
    
    
'''
This function plots the Learning curve and saves to the give directory'''
def plot_learning_curve(history, file_stem):
    
    LEARNING_CURVE_DIR = file_stem + '/Learning-Curve-PointNet-Model'

    if not os.path.exists(LEARNING_CURVE_DIR):
        os.makedirs(LEARNING_CURVE_DIR)

    #want to save figure with the data and time.
    #change to log_scale or power_transform accordingly:
    today_time = str(datetime.today())
    file_name = "log_scale_LC{}{}".format(today_time,'.png')
    

    save_fig_path = os.path.join(LEARNING_CURVE_DIR, file_name)
    plt.plot(history["loss"], label="training loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_fig_path, format='png')
    
'''
This function plots the confusion matrix and saves to the given directory'''
def plot_confusion_matrix(file_stem, label, preds, classes, cmap=plt.cm.Blues):
    
    
    CONFUSION_MATRIX_DIR = file_stem + '/confusion-matrix-directory'
    
    if not os.path.exists(CONFUSION_MATRIX_DIR):
        os.makedirs(CONFUSION_MATRIX_DIR)
    
    today_time = str(datetime.today())
    file_name = "confusion_matrix{}{}".format(today_time,'.png')
    
    save_fig_path = os.path.join(CONFUSION_MATRIX_DIR, file_name)    
    
    cm = confusion_matrix(label, preds)
    cm = np.round(cm/np.sum(cm, axis = 1).reshape(-1,1),2)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    
    plt.savefig(save_fig_path, format='png')

'''
This function converts the .npy arrays that have been train test split into tensorflow dataset for training the model'''

def tf_file_conversion(train_events, test_events, train_labels, test_labels, BATCH_SIZE):
    
    # Converts to Tensorflow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_events, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_events, test_labels))

    # Shuffles the points within each events in the Dataset
    train_dataset = train_dataset.shuffle(len(train_events)).map(augment).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(test_events)).map(augment).batch(BATCH_SIZE)
    
    return train_dataset, test_dataset

@click.command()
@click.argument('file-stem')
def main(file_stem):
    
    train_events = np.load(file_stem + '/Data/TrainTestSplitData/train_events.npy', allow_pickle=True)
    test_events = np.load(file_stem + '/Data/TrainTestSplitData/test_events.npy', allow_pickle=True)
    train_labels = np.load(file_stem + '/Data/TrainTestSplitData/train_labels.npy', allow_pickle=True)
    test_labels = np.load(file_stem + '/Data/TrainTestSplitData/test_labels.npy', allow_pickle=True)
    
    BATCH_SIZE = 32 
    N = 150
    N_classes = 5
    
    # Stores the returned dataset
    train_dataset, test_dataset = tf_file_conversion(train_events, test_events, train_labels, test_labels, BATCH_SIZE)

    # Builds the model
    model = build_point_cloud_model(N, N_classes)
    
    # Fits the model and saves learning curve and confusion matrix
    fit_model(model, train_dataset, test_dataset, file_stem)
    
    
if __name__ == "__main__":
    main()

