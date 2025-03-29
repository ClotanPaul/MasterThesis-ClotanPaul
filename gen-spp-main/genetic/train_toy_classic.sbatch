#!/bin/bash
#SBATCH --job-name=gen_train_toy_classic
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paulioan.clotan@studio.unibo.it
#SBATCH --time=84:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx2080   # or l40 depending on need
#SBATCH --output=output_toy_classic.log
#SBATCH --gres=gpu:1

source /scratch.hpc/paulioan.clotan/MasterThesis-ClotanPaul/gen-spp-main/venv/bin/activate
cd /scratch.hpc/paulioan.clotan/MasterThesis-ClotanPaul/gen-spp-main/genetic
python3 train.py toy toy_experiment_classical --n=5 
