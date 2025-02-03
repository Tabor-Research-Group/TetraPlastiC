#!/bin/bash

## NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=pythonjob        # Set the job name to "lammps"
#SBATCH --time=24:00:00          # Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                      #Request tasks
#SBATCH --cpus-per-task=10     #Request CPUs per task
#SBATCH --mem-per-cpu=7000                

module purge
module load Anaconda3/5.3.0
source activate py3.6
python calc.py
