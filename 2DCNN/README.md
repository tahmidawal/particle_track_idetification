# 2D CNN
## Introduction
In this folder, you will find python and shell scripts relating to reorganizing the `.txt` files from `oldRawDataTxtFiles` into usable formats that can be readable for Neural Network input, as well as a shell script for fitting the training data on a 2D CNN model based on VGG16. There are also Jupyter Notebooks in the `notebooks` folder, which William used for model training. It is fine to use the notebooks, but at a certain point it is not recommended to use them, as training the data on the Jupyter notebooks would take too long. The Jupyter notebooks should be mainly used for plotting, data visualization, and debugging.

## What Each File Does
### src
#### entire_pipeline.sh
This is the main shell script created to turn the .txt files into readable numpy arrays. The shell script calls on four python functions in order. `txt_to_npy_aray.py`, `sum_pads_array.py`, `clean_cat_shuffle.py`, and `data_preprocessing.py`. 

##### txt_to_npy_array.py
This python script reads in the .txt files from `oldRawDataTxtFiles`. It loads in the `.txt` files and turns them into numpy arrays. The key difference between 2D and 3D data processing of the data is that the time data from the `.txt` files is not used. The code slices out the time data that would have been in the numpy arrays. Time data is not used because the 2D CNN structure requires a 2D projection of the data.

##### sum_pads_array.py
Because there might be multiple particle hits on a single detector, the arrays are structured as tuples. The code in this python script sums up the charges in the tuple together. The code loads in the previously reorganized data from `txt_to_npy_array.py`

##### clean_cat_shuffle.py
This python script takes in all the data from the five classes and concatenates them into a single array with labels and data combined.

##### data_preprocessing.py
This is the last python script needed to be run in the shell script `entire_pipeline.sh`. It takes the array organized in `clean_cat_shuffle.py` and performs a train test split on the data, which also separates the inputs from the labels.

#### model_fit.sh (and model_fit.py)
The `model_fit.sh` script only runs one python script, `model_fit.py`, which loads in the train and validation inputs and labels that were generated from the `entire_pipeline.sh` script. `model_fit.sh` then uses VGG16 architecture to train a Convolutional Neural Network to classify the inputs according to their labels. The weights from each epoch will be saved, as well as a `.png` of the learning curve. The model will be saved so one can load in the model and apply the test data to get confusion matrices to visualize how well the data is performing.

### Workflow
To start on the project, one must have the data from `oldRawDataTxtFiles` to start. Check if `oldRawDataTxtFiles` is in the right directory, as where you have it placed may be different to where the python script `txt_to_npy_array.py` in `entire_pipeline.sh` is reading the data from the folder.

Next you want to run `entire_pipeline.sh` so the data can be used as inputs for training neural networks. This might take a little while.

Then you can finally use the train data to train the 2D CNN VGG16 model. Run the shell script `model_fit.sh` and you will find a new folder created in the folder `ModelCheckpoints`.

To get the confusion matrices, use the confusion_matrix.ipynb notebook in the `notebooks` folder, load in the model from the ModelCheckpoints, and run through the cells in the notebook. It should be pretty self explanatory.

### Important Notes
The only notebook you will need to use is the confusion_matrix.ipynb notebook. Feel free to look through the other notebooks, but they will not be able to function properly because the file directories are incorrect.

The files here are mostly all created by William, who worked on the data before I did. Some of the work may be incomplete or unorganized, and some of the directories will likely need to be changed.

William’s results have been proven to work, with decent results for the 2D CNN network that utilizes VGG16 architecture for model training.

### Additional Notes left by William
See the presentation called "VernaCasePresentationpdf.pdf" attached (in doc folder), and the instructions in the "PipelineInstructions.pdf" on how to re-create similar results from the presentation. (i.e. recreating images from pg 23 of the presentation, also in doc folder)

This Github also has the notebook for the recreation of Tom's results, i.e. the confusion matrix from page 16 of "VernaCasePresentationpdf.pdf". Simple put PIDimplementation.ipynb and Simulation.csv in the same directory with the same virtual environment as the single-track project, then run the PIDimplementation.ipynb from top to bottom.


