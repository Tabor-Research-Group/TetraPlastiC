#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=test       #Set the job name
#SBATCH --time=1:00:00 #Set the wall clock limit
#SBATCH --ntasks=1 #4                     #Request tasks
#SBATCH --cpus-per-task=1     #Request CPUs per task
#SBATCH --mem-per-cpu=8000M                       #Request Memory in MB per node
#SBATCH --output=test.%j      #Send stdout/err
sed -i '3i #T wB97XD/def2svpp Volume=tight Geom=AllCheckpoint Guess=Read' test.inp
