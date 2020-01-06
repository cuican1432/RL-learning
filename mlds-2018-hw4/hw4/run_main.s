#!/bin/sh
#SBATCH --job-name=RL-dqn
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=40:00:00


module purge

module load cuda/10.0.130
module load cudnn/9.0v7.0.5
module load anaconda3/4.3.1
source activate py36


# python main.py --train_pg
python main.py --train_dqn