#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]x
#SBATCH -N 1 -c 64 			    # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1		    # Specify number of GPU per task
#SBATCH --ntasks-per-node=1	        # Specify number of GPU cards
#SBATCH --gpus=1                    # Specify total number of GPUs

#SBATCH -t 120:00:00                # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200210             	# Specify project name
#SBATCH -J JOBNAME               	# Specify job name
#SBATCH -o _sbatches_/rungpu_frameselection-%A.out

module purge                        # Unload all modules
module load Miniconda3              # Load the module that you want to use
conda activate /home/wtepsan/Miniconda/wtepsanconda                 # Activate your environment

python3 ROI_train_Joints.py \
--roi_model_name EfficientNetB7 \
--optim_name SGD \
--loss_function_name CrossEntropyLoss \
--num_epoch 80 \
--batch_size 4 \
--benchmark xsub  \
--google_pretrain True \
--transform_resize False \
--evaluation False \
--random_interval True \
--choosing_frame_method btwmax \
--imginput roi5 \
--attention NONE \
--onthefly YES \
--checkcode NO

### --choosing_frame_method option #'randombegin_randomlength', 'random_from_each_interval', 'random'}) 
### --imginput', default="fullbody", choices={'fullbody', 'body2parts', 'body3parts', 'roi3', 'roi4', 'roi5', 'roi6', 'roi7'})