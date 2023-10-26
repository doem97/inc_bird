#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -P 21026844
#PBS -N fo342
#PBS -m be
#PBS -M doem1997@gmail.com
#PBS -o ./exps/nscclog/1026/s7_fo342.log

# Commands start here
module load anaconda3/2022.10
conda activate study
cd ./study/pilot

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 25 Oct 2023, 9:04PM
# **********************************************************************************

python main.py --config="./configs/mpc5/foster_caf1k_384_cs701_lr1e-3_30+40+10_bs16.json" --config_id="foster_caf1k_384_cs701_lr1e-3_30+40+10_bs16"
sleep 20
python inference.py --config="./configs/mpc5/foster_caf1k_384_cs701_lr1e-3_30+40+10_bs16.json" --config_id="foster_caf1k_384_cs701_lr1e-3_30+40+10_bs16"