#!/bin/bash
#SBATCH --job-name "Fit Model Script"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64g
#SBATCH --mail-user=taawal@davidson.edu
#SBATCH --mail-type=END

python python-scripts/Train_test_splitting.py ../Data
