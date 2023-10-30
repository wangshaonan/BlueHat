#!/bin/bash

#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --job-name=phrase

module purge

singularity exec --nv \
	    --overlay /scratch/sw5545/pytorch-example/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python calculate_prob_similarity_between.py"
