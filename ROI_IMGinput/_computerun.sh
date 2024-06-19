#!/bin/bash
#SBATCH -p compute                   # Specify partition [Compute/Memory/GPU] 
#SBATCH --ntasks-per-node=4		    # Specify tasks per node
#SBATCH -t 120:00:00                # Specify maximum time limit (hour: minute: second) 

#SBATCH -A lt200210             	# Specify project name
#SBATCH -J JOBNAME               	# Specify job name
#SBATCH -o ./_sbatches_/runmemory-%A.out

module purge                        # Unload all modules
module load Miniconda3              # Load the module that you want to use
conda activate /home/wtepsan/Miniconda/wtepsanconda     # Activate your environment

python3 ROI_genROI.py --begin 0 --end 124