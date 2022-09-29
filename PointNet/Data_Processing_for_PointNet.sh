#!/bin/bash
#SBATCH --job-name "Fit Model Script"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64g
#SBATCH --gpus 1
#SBATCH --mail-user=taawal@davidson.edu
#SBATCH --mail-type=END

cd /home/DAVIDSON/taawal/Single_Track_Particle_Id/PointNet/Scripts

source /opt/conda/bin/activate SPIRIT

python Data_Processing_for_PointNet.py ffjhj
