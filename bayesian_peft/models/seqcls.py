import torch.nn as nn
from run import get_modelwrapper
import torch as t
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftConfig,
)


class SeqCls(nn.Module):
    def __init__(self, args, accelerator=None, tokenizer=None, **kwargs) -> None:
        super().__init__()
        self.args = args
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if args.load_checkpoint:
            print('=====================')
            args.load_path = f'checkpoints/{args.model_type}_{args.modelwrapper}/{args.model}/{args.dataset}/{args.load_model_path}'
            print('Loading model from: ', args.load_path)
            peft_config = PeftConfig.from_pretrained(args.load_path, is_trainable=False)
            model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
            self.model = PeftModel.from_pretrained(model, args.load_path, is_trainable=False)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            self.model.print_trainable_parameters()
            print('Model loaded successfully')
            print('=====================')
        else:

            print('++++++++++++++++++++++++++++++++++++++++')
            print('Load Tokenizer:', os.path.join('../my_llm', args.model))
            print('++++++++++++++++++++++++++++++++++++++++')

            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join('../my_llm', args.model),
                use_fast=not args.use_slow_tokenizer,
                trust_remote_code=True
            )


            if args.load_model_path is not None:
                model = AutoModelForSequenceClassification.from_pretrained(args.load_model_path, num_labels=args.outdim)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.outdim)
            # if args.apply_classhead_lora:
            #     target_modules = ["q_proj", "v_proj", "lm_head"]
            # elif args.apply_qkv_head_lora:
            #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
            # else:
            #     target_modules = ["q_proj", "v_proj"]
            target_modules = ["query", "value"]

            if not args.ood_ori_dataset:
                peft_config = LoraConfig(inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha,
                                         lora_dropout=args.lora_dropout, target_modules=target_modules)
            else:
                print('+++++++++++++++model eval+++++++++++++++')
                peft_config = LoraConfig(inference_mode=True, r=args.lora_r, lora_alpha=args.lora_alpha,
                                         lora_dropout=args.lora_dropout, target_modules=target_modules)
            self.model = get_peft_model(model, peft_config)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")

            if self.model.config.pad_token_id == None:
                self.model.config.pad_token_id = tokenizer.eos_token_id

            self.model.print_trainable_parameters()
