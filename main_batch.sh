#!/bin/sh
# embedded options to bsub - start with #BSUB
# -- Name of the job ---
#BSUB -J ais_normality_model
# -- specify queue --
#BSUB -q gpua100
### -- ask for number of cores (default: 1) -- 
#BSUB -n 8
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=2GB]"
### -- specify we want a gpu with 32GB -- 
###BSUB -R "select[gpu32gb]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 4GB
### -- set walltime limit: hh:mm -- Maximum of 24 hours --
#BSUB -W 71:45 
### -- user email address --
#BSUB -u s192309@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
### -- end of LSF options --

# here follow the commands you want to execute

# Unload already installed software
module unload cuda
module unload cudnn
# load modules
module load python3/3.9.6
module load cuda/11.5 
module load cudnn/v8.3.0.98-prod-cuda-11.5 
# activate the virtual environment which includes the necessary python packages
source ./python_ais_env_latest/bin/activate

# run program
python src/models/train_evaluate_Bornholm_4.py

