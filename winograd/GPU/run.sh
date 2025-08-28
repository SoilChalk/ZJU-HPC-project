#!/bin/bash
#SBATCH --job-name=winograd
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --time=00:10:00
#SBATCH --partition=V100

./winograd inputs/config.txt
