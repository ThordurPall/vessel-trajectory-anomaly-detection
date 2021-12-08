#!/bin/sh

# Stop on error
set -e

# Unload already installed software
module unload cuda
module unload cudnn

# load modules
module load python3/3.6.13
module load cuda/11.1
module load cudnn/v8.0.4.30-prod-cuda-11.1

# setup virtual environment
python3 -m venv python_ais_env
source ./python_ais_env/bin/activate

# install needed packages
pip3 install -U -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
