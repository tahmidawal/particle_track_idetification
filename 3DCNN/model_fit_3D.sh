#!/bin/bash
#SBATCH --job-name "3D Model Fit Script"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120g
#SBATCH --gpus 1
#SBATCH --mail-user=iaheung@davidson.edu
#SBATCH --mail-type=END

cd /home/DAVIDSON/iaheung/spirit/Single_Track_Particle_Id/3DCNN/src

source /opt/conda/bin/activate spirit
cd .. 
cd scripts

python model_fit_3D.py