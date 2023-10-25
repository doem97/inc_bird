#!/bin/bash

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 10 Oct 2023
# **********************************************************************************

export CUDA_VISIBLE_DEVICES="6"
# python main.py --config="./configs/foster.json"
# python main.py --config="./configs/foster_cub.json"
# python main.py --config="./configs/foster_cs701_lr1e-3_mem2000_mpc20_im21k.json" --config_id="foster_cs701_lr1e-3_mem2000_mpc20_im21k"
python main.py --config="./configs/foster_caf22ft1k_384_cs701_lr1e-3+20_lr1e-3+25_mpc25_bs16.json" --config_id="foster_caf22ft1k_384_cs701_lr1e-3+20_lr1e-3+25_mpc25_bs16"