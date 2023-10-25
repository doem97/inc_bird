#!/bin/bash

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 10 Oct 2023
# **********************************************************************************

export CUDA_VISIBLE_DEVICES="0"
# python main.py --config="./configs/foster_cubcomb_lr1e-3_mem2000_mpc20_bs16_im21k.json" --config_id="foster_cubcomb_lr1e-3_mem2000_mpc20_bs16_im21k"
# python inference.py --config="./configs/foster_cubcomb_lr1e-3_mem2000_mpc20_bs16_im21k.json" --config_id="foster_cubcomb_lr1e-3_mem2000_mpc20_bs16_im21k"
# python main.py --config="./configs/foster_cs701_lr1e-3_mem2000_mpc20_bs16_15e_im21k.json" --config_id="foster_cs701_lr1e-3_mem2000_mpc20_bs16_15e_im21k"
# python main.py --config="./configs/foster_cs701_lr1e-3_mem2000_mpc20_bs16_30e_im21k.json" --config_id="foster_cs701_lr1e-3_mem2000_mpc20_bs16_30e_im21k"
python main.py --config="./configs/foster_cs701_lr1e-3+5e-4_mpc25_bs64_im21k.json" --config_id="foster_cs701_lr1e-3+5e-4_mpc25_bs64_im21k"