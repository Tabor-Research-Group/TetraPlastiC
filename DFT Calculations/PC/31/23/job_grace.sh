#!/bin/bash
#SBATCH -J confgen          # job name
#SBATCH -o confgen.out
#SBATCH -e confgen.err

#SBATCH -N 1            # number of nodes requested
#SBATCH -n 10           # total number of mpi tasks requested
#SBATCH -t 12:00:00     # run time (hh:mm:ss) - 0.1 hours
#SBATCH --mem-per-cpu=6000

CUR_DIR=$(pwd)
RESULTS=$(pwd)

ml Anaconda3/2020.07
source activate xtb
export OMP_STACKSIZE=6G
#export OMP_NUM_THREADS=4

WORK_DIR=/scratch/user/`whoami`/`whoami`/${SLURM_JOB_ID}_${clock}
echo "Setting up directories"
echo "  at ${WORK_DIR}"
mkdir -p $WORK_DIR
cp -R $CUR_DIR/* $WORK_DIR
cd $WORK_DIR

echo Running conformer job in $(pwd)
crest out.xyz --gfn 2 --chrg 0 -T 1 > confgen.log
wait

cp $WORK_DIR/*.log $RESULTS/
cp $WORK_DIR/*.xyz $RESULTS/
cp $WORK_DIR/*.out $RESULTS/
cp $WORK_DIR/*.err $RESULTS/
echo 'Cleaning up'
rm -R $WORK_DIR
cd $CUR_DIR

num_atom=$(awk 'NR==1 {print $1}' crest_conformers.xyz)
num_conf=$(($(cat crest_conformers.xyz| wc -l)/$(($num_atom+2))))

function split_conf {
    split -l $(($num_atom+2)) --numeric-suffixes=1 $1.xyz Conf_ --additional-suffix=.xyz
    }
if (($num_conf > 20)) 
then
    head -$((($num_atom+2)*20)) crest_conformers.xyz > tmp.xyz
    split_conf tmp
    rm tmp.xyz
else
    split_conf crest_conformers
fi

for file in Conf_0*.xyz
do
    mv "$file" $(echo $file | sed -E "s/^Conf_0/Conf_/")
done

