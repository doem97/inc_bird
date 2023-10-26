#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -P 21026844
#PBS -N foster_caf1k_384_cs701_lr1e-3_bs24
#PBS -m be
#PBS -M doem1997@gmail.com
#PBS -o ./exps/nscclog/nscc_output_s12.log

# Commands start here
module load anaconda3/2022.10
conda activate study
cd ./study/pilot

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 25 Oct 2023, 9:04PM
# **********************************************************************************

python main.py --config="./configs/aft/foster_caf1k_384_cs701_lr1e-3_bs24.json" --config_id="foster_caf1k_384_cs701_lr1e-3_bs24"
sleep 20
python inference.py --config="./configs/aft/foster_caf1k_384_cs701_lr1e-3_bs24.json" --config_id="foster_caf1k_384_cs701_lr1e-3_bs24"