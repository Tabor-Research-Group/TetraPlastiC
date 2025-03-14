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
    for ((i=1;i<=100;i++))
    do
        if [ $i == 2 ]
        then
            sed -i '4d' gaussian_test.inp
            sed -i '4i #T wB97XD//def2svpp Volume=tight Geom=AllCheckpoint Guess=Read' gaussian_test.inp
            sed -n "1,$(( 8 - 1 )) p; 8 q" gaussian_test.inp > tmp.inp
            mv tmp.inp gaussian_test.inp
        fi
        g16 $INPUT_FILE > $OUTPUT_FILE
        grep Molar $OUTPUT_FILE >> volume.out
    done
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
    cp $WORK_DIR/volume.out $RESULTS/
    echo 'Cleaning up'
    cd $CUR_DIR
    rm -R $WORK_DIR
    }

    trap 'kill -9 $pid; clean_up; exit' SIGTERM SIGINT
    time_run

    wait
    clean_up
else
    time_run

    wait
fi


