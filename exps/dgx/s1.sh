#!/bin/bash

# **********************************************************************************
# **Inference** on fine-tune checkpoints with different steps - 256_1000ep
# Date: 10 Oct 2023
# **********************************************************************************

export CUDA_VISIBLE_DEVICES="4"
# python main.py --config="./configs/foster.json"
# python main.py --config="./configs/foster_cub.json"
python main.py --config="./configs/foster_cs701.json" --config_id="foster_cs701_raw"