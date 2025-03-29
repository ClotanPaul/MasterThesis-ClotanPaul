#!/bin/bash
#SBATCH --job-name=genetic_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paulioan.clotan@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx2080   # or l40 depending on need
#SBATCH --output=training_output.log
#SBATCH --gres=gpu:1

source /scratch.hpc/paulioan.clotan/MasterThesis-ClotanPaul/gen-spp-main/venv/bin/activate
cd /scratch.hpc/paulioan.clotan/MasterThesis-ClotanPaul/gen-spp-main/genetic
python3 train.py toy toy_experiment --n=2
