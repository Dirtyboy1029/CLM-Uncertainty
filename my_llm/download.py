# -*- coding: utf-8 -*- 
# @Time : 2025/9/25 20:09 
# @Author : DirtyBoy 
# @File : download.py

import os
# logging.set_verbosity_info()
#
# # 设置 Hugging Face 镜像
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import os

##  export HF_ENDPOINT=https://hf-mirror.com

# 模型名称 Salesforce/codegen-6B-multi  deepseek-ai/deepseek-coder-6.7b-base

cache_dir = "./hf_cache"

model_name = "Salesforce/codegen-6B-multi"
save_path = "./Salesforce/codegen-6B-multi"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, )
model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, cache_dir=cache_dir, )
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
