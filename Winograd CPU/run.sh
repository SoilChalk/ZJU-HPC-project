#!/bin/bash
#SBATCH --job-name=winograd_final_optimized
#SBATCH --partition=kunpeng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Compiling final optimized version from src/ directory..."
make clean
make

echo "Running final benchmark..."
./winograd ./inputs/config.txt

echo "Moving final log to results/ directory..."
mv winograd_final_optimized_${SLURM_JOB_ID}.log results/final_optimized.log

echo "Done."