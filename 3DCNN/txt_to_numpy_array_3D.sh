#!/bin/bash
#SBATCH --job-name "3D txt-np"
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64g
#SBATCH --mail-user=anjones1@davidson.edu
#SBATCH --mail-type=END



#Commands for converting text files to numpy arrays
python scripts/txt_to_npy_array_3D.py --txtfilename=../Data/oldRawDataTxtFiles/output_d.txt --npyfilename=../Data/3DsmallDataParticleChargeArrays/deuteron_array_3D.npy
python scripts/txt_to_npy_array_3D.py --txtfilename=../Data/oldRawDataTxtFiles/output_He3.txt --npyfilename=../Data/3DsmallDataParticleChargeArrays/he3_array_3D.npy
python scripts/txt_to_npy_array_3D.py --txtfilename=../Data/oldRawDataTxtFiles/output_He4.txt --npyfilename=../Data/3DsmallDataParticleChargeArrays/he4_array_3D.npy
python scripts/txt_to_npy_array_3D.py --txtfilename=../Data/oldRawDataTxtFiles/output_p.txt --npyfilename=../Data/3DsmallDataParticleChargeArrays/proton_array_3D.npy
python scripts/txt_to_npy_array_3D.py --txtfilename=../Data/oldRawDataTxtFiles/output_t.txt --npyfilename=../Data/3DsmallDataParticleChargeArrays/triton_array_3D.npy

#converts numpy arrays into usable 3D formats
python scripts/npy_array_to_3Ddata.py
