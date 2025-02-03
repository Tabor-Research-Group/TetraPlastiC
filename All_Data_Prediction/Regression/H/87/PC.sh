#!/bin/bash

##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION

#SBATCH --mem=30Gb                 #Request Memory in MB per node

#SBATCH -t 10:00:00                 #Time for the job to run
#SBATCH -J PC          #Name of the job
#SBATCH -e PC.err

#SBATCH -N 1                    #Number of nodes required
#SBATCH -n 10                    #Number of cores needed for the job

module load intel/2017a

export mpiranks=1
export OMP_NUM_THREADS=10


mpirun -np 10 SISSO_3.3 > log
