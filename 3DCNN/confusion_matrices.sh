#!/bin/bash
#SBATCH --job-name "confusion matrix"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64g
#SBATCH --gpus 1
#SBATCH --mail-user=iaheung@davidson.edu
#SBATCH --mail-type=END

# ACTIVATE PYTHON VIRTUAL ENVIRONMENT HERE
cd /home/DAVIDSON/iaheung/spirit/Single_Track_Particle_Id/3DCNN/src

source /opt/conda/bin/activate spirit
cd .. 
cd scripts

python confusion_matrices.py