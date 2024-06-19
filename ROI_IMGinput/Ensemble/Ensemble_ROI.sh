#!/bin/bash
#SBATCH -p compute                   # Specify partition [Compute/Memory/GPU] 
#SBATCH --ntasks-per-node=4		    # Specify tasks per node
#SBATCH -t 120:00:00                # Specify maximum time limit (hour: minute: second) 

#SBATCH -A lt200210             	# Specify project name
#SBATCH -J JOBNAME               	# Specify job name
#SBATCH -o ENNSEMBLE-%A.out

module purge                        # Unload all modules
module load Mamba/23.11.0-0         # Load the module that you want to use
conda activate wtepsan_env     # Activate your environment

# python3 Ensemble_ROI.py --protocols xview

# python3 Ensemble_ROI_3IMG.py --protocols xsub
# python3 Ensemble_ROI_3IMG.py --protocols xview

# python3 Ensemble_ntu60_LAGCN.py --protocols xsub
# python3 Ensemble_ntu60_LAGCN.py --protocols xview
# python3 Ensemble_ntu60_LAGCN_2RGB.py --protocols xsub
# python3 Ensemble_ntu60_LAGCN_2RGB.py --protocols xview

python3 Ensemble_ntu60_MMNet.py --protocols xsub
# python3 Ensemble_ntu60_LAGCN.py --protocols xview
python3 Ensemble_ntu60_MMNet_2RGB.py --protocols xsub
# python3 Ensemble_ntu60_LAGCN_2RGB.py --protocols xview