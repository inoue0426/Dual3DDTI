#!/bin/bash -l     
#SBATCH --time=96:00:00
#SBATCH --mem=20gb
#SBATCH --requeue
#SBATCH --job-name='pdb gcn'
#SBATCH --partition a100-4 
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=inoue019@umn.edu 

module load conda
source activate py39
python pdb_gt.py