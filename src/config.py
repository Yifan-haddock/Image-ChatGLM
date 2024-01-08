# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:29
# @author  : Mo
# @function: config of chatglm2


# optimized for RTX 4090. for larger GPUs, increase some of these?
# MICRO_BATCH_SIZE = 4  # default=4  # this could actually be 5 but i like powers of 2
# MICRO_BATCH_SIZE = 16  # default=4  # this could actually be 5 but i like powers of 2
MICRO_BATCH_SIZE = 32  # default=4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4  # default=3e-4  # the Karpathy constant
EPOCHS = 3  # default=3  # we don't always need 3 tbh
# LORA_DROPOUT = 0.1
# LORA_ALPHA = 32
# LORA_R = 32
LORA_DROPOUT = 0.05
LORA_ALPHA = 16
LORA_R = 8
SAVE_STEPS = 512
VAL_SET_SIZE = 0
MAX_LENGTH_Q = 256 - 2  # default=256 - 2
MAX_LENGTH_A = 256 - 2  # default=256 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 4
TUNING_MODULE = 'all'

PATH_MODEL_PRETRAIN = "/Share/home/qiyifan/filebase/source/chatglm2-6b"
PATH_MODEL_IMAGE = "/Share/home/qiyifan/filebase/source/beit-base-patch16-224-pt22k-ft22k"
REPO_ID = "THUDM/chatglm2-6b"
PATH_MODEL_PRETRAIN = PATH_MODEL_PRETRAIN if PATH_MODEL_PRETRAIN else REPO_ID
MODEL_SAVE_DIR = "../tmp/IMAGEGLM_ALL"

IS_PARALLELIZABLE = False
MODEL_PARALLEL = False
USE_CACHE = False
CUDA_VISIBLE_DEVICES = "0"
USE_TORCH = "1"
CPU_NUMS = "9"
USE_CUDA = False if CUDA_VISIBLE_DEVICES == "-1" else True
PROMPT_EXPLAIN = False