#!/bin/bash
#SBATCH --partition=cosmos
#SBATCH --nodelist=blazar
#SBATCH --time=17:00:00
#SBATCH --gres=gpu:1,gpu_mem:7100
#SBATCH --mem=6000
#SBATCH --qos=intermediate

source activate my_venv_2_7
#python eval.py
#python train.py --hidden_dim 10
#python train.py --hidden_dim 20
#python train.py --hidden_dim 30
python train.py --projection=True --hidden_dim=250 --name=5b > result_5b

