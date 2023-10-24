#!/bin/bash

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 10 Oct 2023
# **********************************************************************************

export CUDA_VISIBLE_DEVICES="3"
python main.py --config="./configs/foster_cs701_lr1e-3_mem2000_mpc20_im21k.json" --config_id="foster_cs701_lr1e-3_mem2000_mpc20_im21k"