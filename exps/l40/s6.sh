#!/bin/bash

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 10 Oct 2023
# **********************************************************************************

export CUDA_VISIBLE_DEVICES="3"
python main.py --config="./configs/foster_cs701_lr8e-4_mem2000_mpc20_bs16_im21k.json" --config_id="foster_cs701_lr8e-4_mem2000_mpc20_bs16_im21k"