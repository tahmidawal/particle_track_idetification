#!/bin/bash
#SBATCH --job-name "Fit Model Script"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64g
#SBATCH --gpus 1
#SBATCH --mail-user=iaheung@davidson.edu
#SBATCH --mail-type=END

cd /home/DAVIDSON/iaheung/Single_Track_Particle_Id/2DCNN

source /opt/conda/bin/activate spirit

python model_fit.py