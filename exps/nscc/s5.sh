#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -P 21026844
#PBS -N dso
#PBS -m be
#PBS -M doem1997@gmail.com
#PBS -o ./exps/nscclog/nscc_output_s5.log

# Commands start here
module load anaconda3/2022.10
conda activate study
cd ./study/pilot

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 10 Oct 2023
# **********************************************************************************

# python main.py --config="./configs/foster_cubcomb_lr1e-3_mem2000_mpc20_bs16_im21k.json" --config_id="foster_cubcomb_lr1e-3_mem2000_mpc20_bs16_im21k"
# python inference.py --config="./configs/foster_cubcomb_lr1e-3_mem2000_mpc20_bs16_im21k.json" --config_id="foster_cubcomb_lr1e-3_mem2000_mpc20_bs16_im21k"
# python main.py --config="./configs/foster_cs701_lr1e-3_mem2000_mpc20_bs16_15e_im21k.json" --config_id="foster_cs701_lr1e-3_mem2000_mpc20_bs16_15e_im21k"
# python main.py --config="./configs/foster_cs701_lr1e-3_mem2000_mpc20_bs16_30e_im21k.json" --config_id="foster_cs701_lr1e-3_mem2000_mpc20_bs16_30e_im21k"
python inference.py --config="./configs/big_sample/foster_caf22ft1k_384_cs701_lr1e-3_mpc25_bs16.json" --config_id="foster_caf22ft1k_384_cs701_lr1e-3_mpc25_bs16" &
python inference.py --config="./configs/big_sample/foster_caf22ft1k_384_cs701_lr1e-3+20_lr1e-3+25_mpc25_bs16.json" --config_id="foster_caf22ft1k_384_cs701_lr1e-3+20_lr1e-3+25_mpc25_bs16" &
python inference.py --config="./configs/big_sample/foster_caf22ft1k_384_cs701_lr1e-3+20_lr1e-3+25_mpc25_bs16.json" --config_id="foster_caf22ft1k_384_cs701_lr1e-3+30_lr1e-3+25_mpc25_bs16" &
python inference.py --config="./configs/big_sample/foster_caf22ft1k_384_cs701_lr1e-3+20_lr1e-3+30_mpc25_bs16.json" --config_id="foster_caf22ft1k_384_cs701_lr1e-3+20_lr1e-3+30_mpc25_bs16"

wait