#!/bin/bash

##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION

#SBATCH --mem=20Gb                 #Request Memory in MB per node

#SBATCH -t 12:00:00                 #Time for the job to run
#SBATCH -J Tm          #Name of the job
#SBATCH -e Tm.err

#SBATCH -N 1                    #Number of nodes required
#SBATCH -n 3                    #Number of cores needed for the job

module load Anaconda3/2021.05
source activate py3.6

python Training_ex_Tm.py
