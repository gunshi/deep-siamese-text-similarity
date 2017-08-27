#!/bin/bash
#SBATCH --partition=cosmos
#SBATCH --nodelist=quasar
#SBATCH --time=17:59:00
#SBATCH --gres=gpu:1,gpu_mem:5000
#SBATCH --mem=6000
#SBATCH --qos=intermediate

source activate my_venv_2_7
python train.py --return_outputs False --hidden_dim=50 --name=6a --lr 0.00001 > result_6a
#python train.py --return_outputs False --hidden_dim=50 --name=2a --lr 0.00001 > result_2a
#python train.py --return_outputs True  --hidden_dim=50 --name=2b --lr 0.00001 > result_2b
#python train.py --return_outputs True  --hidden_dim=50 --name=2c --lr 0.0001 > result_2c
#python train.py --return_outputs True  --hidden_dim=50 --name=2d --lr 0.00001 --conv_net_training True > result_2d
#python train.py --return_outputs True  --hidden_dim=50 --name=2e --lr 0.00001 --conv_net_training True --num_lstm_layers 3 > result_2e

