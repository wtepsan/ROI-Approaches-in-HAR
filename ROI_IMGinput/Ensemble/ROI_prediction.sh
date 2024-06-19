#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]x
#SBATCH -N 1 -c 16   			    # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1		    # Specify number of GPU per task
#SBATCH --ntasks-per-node=1	        # Specify number of GPU cards
#SBATCH --gpus=1                    # Specify total number of GPUs

#SBATCH -t 120:00:00                # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200048               	# Specify project name
#SBATCH -J JOBNAME               	# Specify job name
#SBATCH -o Ensemble/rungpu-%A.out

module purge                        # Unload all modules
module load Miniconda3              # Load the module that you want to use
conda activate /home/wtepsan/Miniconda/wtepsanconda                 # Activate your environment

# python3 ROI_prediction.py --roi_model_name EfficientNetB7 --optim_name SGD --loss_function_name CrossEntropyLoss --num_epoch 1 --batch_size 8 --benchmark xview  --google_pretrain True --resize NO --attention NONE #--checkcode YES
# python3 ROI_prediction.py --roi_model_name EfficientNetB7 --optim_name SGD --loss_function_name CrossEntropyLoss --num_epoch 1 --batch_size 8 --benchmark xview  --google_pretrain True --resize NO --attention OpticalFlow
# python3 ROI_prediction.py --roi_model_name EfficientNetB7 --optim_name SGD --loss_function_name CrossEntropyLoss --num_epoch 1 --batch_size 8 --benchmark xsub  --google_pretrain True --resize NO --attention NONE
python3 ROI_prediction.py --roi_model_name EfficientNetB7 --optim_name SGD --loss_function_name CrossEntropyLoss --num_epoch 1 --batch_size 8 --benchmark xsub  --google_pretrain True --resize NO --attention OpticalFlow