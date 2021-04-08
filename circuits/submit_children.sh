#!/bin/bash
#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=Children
# %x=job-name %j=jobid
#SBATCH --output=%x_%j.out
cd ${SLURM_SUBMIT_DIR}
echo "Starting sbatch script myscript.sh at:`date`"
echo "  running host:    ${SLURMD_NODENAME}"
echo "  assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "  jobid:           ${SLURM_JOBID}"
# show me my assigned GPU number(s):
echo "  GPU(s):          ${CUDA_VISIBLE_DEVICES}"

python submit_children.py $1
