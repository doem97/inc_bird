#!/bin/bash

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 10 Oct 2023
# **********************************************************************************

export CUDA_VISIBLE_DEVICES="5"
# python main.py --config="./configs/foster.json"
# python main.py --config="./configs/foster_cub.json"
# python main.py --config="./configs/foster_cs701_lr8e-4_mem1800_mpc20.json" --config_id="foster_cs701_lr8e-4_mem1800_mpc20"
python main.py --config="./configs/foster_caf_cubcomb_lr1e-3_mpc25_bs16_im21k.json" --config_id="foster_caf_cubcomb_lr1e-3_mpc25_bs16_im21k"