#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JCJNVHVKDPMDAT-UHFFFAOYNA-N_wB97XD_def2svvp_gaussian       #Set the job name
#SBATCH --time=23:59:00 #Set the wall clock limit
#SBATCH --ntasks=1 #4                     #Request tasks
#SBATCH --cpus-per-task=4     #Request CPUs per task
#SBATCH --mem-per-cpu=8000M                       #Request Memory in MB per node
#SBATCH --output=gaussian_job.%j      #Send stdout/err

##OPTIONAL JOB SPECIFICATIONS #SBATCH --account=123456 #Set billing account to 123456
ml purge
ml Gaussian/g16_B01
SCRATCH=/scratch/user/`whoami`

# Settings
INPUT_FILE="gaussian_test.inp"
OUTPUT_FILE="gaussian_test.log"
CUR_DIR=$(pwd)
RESULTS=$(pwd)

echo "" >> gaussian_test.inp

#export g16root=/sw/group/lms/sw/g16_B01/   #set g09root variable
#. $g16root/g16/bsd/g16.profile         #source g09.profile to setup environment for g09

time_run(){
{ time -p g16 $INPUT_FILE > $OUTPUT_FILE & pid=$!; } 2>> $OUTPUT_FILE

}

if [ -d $SCRATCH ] && [[ $1 != noscratch ]]; then
    #setup scratch dirs
    clock=`date +%s%15N`
    WORK_DIR=$SCRATCH/$USER/${SLURM_JOBID}_${clock}
    echo "Setting up directories"
    echo "  at ${WORK_DIR}"
    mkdir -p $WORK_DIR
    cp -R $CUR_DIR/* $WORK_DIR
    cd $WORK_DIR

    clean_up()
    {
    # move results of calc
    touch job_manager-complete
#    echo "Copying results back to ${RESULTS}"
#    rsync -a $WORK_DIR/* $RESULTS/ --exclude=slurm-*
    cp $WORK_DIR/job_manager-complete $RESULTS/
    cp $WORK_DIR/*.log $RESULTS/
    cp $WORK_DIR/*.chk $RESULTS/
    echo 'Cleaning up'
    cd $CUR_DIR
    rm -R $WORK_DIR
    # Remove scripting files we are not using, lower file counts
    rm -f job_lsf_ada.sh
    rm -f job_vici.sh
    rm -f lsf_curie.sh
    }

    trap 'kill -9 $pid; clean_up; exit' SIGTERM SIGINT

    time_run

    wait
    clean_up
else
    time_run

    wait
fi


