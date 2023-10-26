#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -P 21026844
#PBS -N der_higherep
#PBS -m be
#PBS -M doem1997@gmail.com
#PBS -o ./exps/nscclog/1026/s2_derhighep.log

# Commands start here
module load anaconda3/2022.10
conda activate study
cd ./study/pilot

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 25 Oct 2023, 9:04PM
# **********************************************************************************

python main.py --config="./configs/aft/der_cs701_caf1k_bs24_30+30.json" --config_id="der_cs701_caf1k_bs24_30+30"
sleep 20
python inference.py --config="./configs/aft/der_cs701_caf1k_bs24_30+30.json" --config_id="der_cs701_caf1k_bs24_30+30"