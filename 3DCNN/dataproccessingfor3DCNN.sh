#!/bin/bash
#SBATCH --job-name "Numpy Arrays to 3D CNN inputs"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 100g
#SBATCH --mail-user=iaheung@davidson.edu
#SBATCH --mail-type=END

cd /home/DAVIDSON/iaheung/spirit/Single_Track_Particle_Id/3DCNN/src

source /opt/conda/bin/activate spirit

cd .. 
cd scripts

#loads in the data created by txt_to_numpy_array_3D.sh, and turns them into a format readable by a 3D CNN
python npy_array_to_3Dcnndata.py
#peforms a train-validation-test split on the 3D CNN data
python train_test_split.py