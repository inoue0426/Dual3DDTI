#!/bin/bash -l     
#SBATCH --time=96:00:00
#SBATCH --mem=400gb
#SBATCH --requeue
#SBATCH --job-name='cmap gat'
#SBATCH --partition a100-4 
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=inoue019@umn.edu 

module load conda
source activate py39
python cmap_GCNConv.py
