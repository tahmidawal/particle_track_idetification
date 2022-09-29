#!/bin/bash
#SBATCH --job-name "Fit Model Script"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64g
#SBATCH --gpus 1
#SBATCH --mail-user=phgabra@davidson.edu
#SBATCH --mail-type=END

python python-scripts/Model_build_and_fit.py ..
