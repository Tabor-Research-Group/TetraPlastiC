#!/bin/bash
#SBATCH -J qchem -e qchem.job.e.%j -o qchem.job.o.%j
#SBATCH -t 24:00:00
#SBATCH -n 8
#SBATCH --mem=8G

module purge
ml qchem/5.3.2-openmp

# Settings
INPUT_FILE="qchem_test.inp"
OUTPUT_FILE="qchem_test.out"
CUR_DIR=$(pwd)
RESULTS=$(pwd)
OMP_NUM_THREADS=8
clock=`date +%s%15N`
export QCSCRATCH=$SCRATCH/$USER/${SLURM_JOBID}_${clock}

time_run(){
{ time -p  qchem -nt $OMP_NUM_THREADS $INPUT_FILE > $OUTPUT_FILE & pid=$!; } 2>> $OUTPUT_FILE
}

if [ -d $SCRATCH ] && [[ $1 != noscratch ]]; then
    #setup scratch dirs
    #clock=`date +%s%15N`
    WORK_DIR=$SCRATCH/$USER/${SLURM_JOBID}_${clock}
    echo "Setting up directories"
    echo "  at ${WORK_DIR}"
    mkdir -p $WORK_DIR
    cp -R $CUR_DIR/* $WORK_DIR
    cd $WORK_DIR

    clean_up()
    {

    cp $WORK_DIR/* $RESULTS/

    echo 'Cleaning up'
    cd $CUR_DIR
    rm -R $WORK_DIR
    # Remove scripting files we are not using, lower file counts
    }

    trap 'kill -9 $pid; clean_up; exit' SIGTERM SIGINT

    time_run

    wait
    clean_up
else
    time_run

    wait
fi

