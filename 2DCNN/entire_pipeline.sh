#!/bin/bash
#SBATCH --job-name "Python Script"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64g
#SBATCH --mail-user=iaheung@davidson.edu
#SBATCH --mail-type=END

# ACTIVATE PYTHON VIRTUAL ENVIRONMENT HERE
cd /home/DAVIDSON/iaheung/spirit/Single_Track_Particle_Id/2DCNN/src

source /opt/conda/bin/activate spirit
cd .. 
cd scripts

#Commands for converting text files to numpy arrays
python txt_to_npy_array.py --npyfilename=deuteron_array.npy
python txt_to_npy_array.py --npyfilename=he3_array.npy
python txt_to_npy_array.py --npyfilename=he4_array.npy
python txt_to_npy_array.py --npyfilename=proton_array.npy
python txt_to_npy_array.py --npyfilename=triton_array.npy

#Commands for summing:
python sum_pads_arrays.py --npyfilename=deuteron_array.npy --output=deuteron.npy
python sum_pads_arrays.py --npyfilename=he3_array.npy --output=he3.npy
python sum_pads_arrays.py --npyfilename=he4_array.npy --output=he4.npy
python sum_pads_arrays.py --npyfilename=proton_array.npy --output=proton.npy
python sum_pads_arrays.py --npyfilename=triton_array.npy --output=triton.npy

#commands for all events creator:
python clean_cat_shuffle.py --he3=he3.npy  --p=proton.npy --t=triton.npy --d=deuteron.npy --he4=he4.npy

#commands for data preprocessing:
python data_preprocessing.py --npy_input_filename=all_events_all_particles_array_shuffled.npy --power_transform=0 --log_scale=1