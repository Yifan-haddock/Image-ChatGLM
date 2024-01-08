from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import (get_peft_model_state_dict, get_peft_model, LoraConfig)
from typing import Optional
from transformers import BeitModel, BeitImageProcessor
import wandb
from datasets import load_dataset
import datasets
import torch.nn as nn
import transformers
import torch
import os
import copy
import importlib
from custom_trainer import CustomTrainer
import matplotlib.pyplot as plt

from modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from tokenization_chatglm import ChatGLMTokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-cf","--config_file")
package = parser.parse_args().config_file

PATH_MODEL_PRETRAIN = getattr(__import__(package, fromlist=[None]),  "PATH_MODEL_PRETRAIN")
PATH_MODEL_IMAGE = getattr(__import__(package, fromlist=[None]),  "PATH_MODEL_IMAGE")
MODEL_SAVE_DIR = getattr(__import__(package, fromlist=[None]),  "MODEL_SAVE_DIR")

MICRO_BATCH_SIZE = getattr(__import__(package, fromlist=[None]),  "MICRO_BATCH_SIZE")
GRADIENT_ACCUMULATION_STEPS = getattr(__import__(package, fromlist=[None]),  "GRADIENT_ACCUMULATION_STEPS")

LEARNING_RATE = getattr(__import__(package, fromlist=[None]),  "LEARNING_RATE")
EPOCHS = getattr(__import__(package, fromlist=[None]),  "EPOCHS")
SAVE_STEPS = getattr(__import__(package, fromlist=[None]),  "SAVE_STEPS")
TUNING_MODULE = getattr(__import__(package, fromlist=[None]),  "TUNING_MODULE")

USE_CACHE = getattr(__import__(package, fromlist=[None]),  "USE_CACHE")

MAX_LENGTH_Q = getattr(__import__(package, fromlist=[None]),  "MAX_LENGTH_Q")
MAX_LENGTH_A = getattr(__import__(package, fromlist=[None]),  "MAX_LENGTH_A")
MAX_LENGTH_QA = getattr(__import__(package, fromlist=[None]),  "MAX_LENGTH_QA")

LORA_DROPOUT = getattr(__import__(package, fromlist=[None]),  "LORA_DROPOUT")
LORA_ALPHA = getattr(__import__(package, fromlist=[None]),  "LORA_ALPHA")
LORA_R = getattr(__import__(package, fromlist=[None]),  "LORA_R")

ID_MASK = 64789
ID_gMASK = 64790
ID_sMASK = 64791
ID_SOP = 64792
ID_EOP = 64793
ID_BOS = 1
ID_EOS = 2
ID_PAD = 0
WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"

def find_target_modules(model, tuning_module = "all"):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if tuning_module == 'all':
            if "Linear" in str(type(module)) and "align" not in name:
                # Extract the Type of the Layer
                layer_type = name.split('.')[-1]
                
                # Add the Layer Type to the Set of Unique Layers
                unique_layers.add(name)
        elif tuning_module == 'text_module':
            if "Linear" in str(type(module)) and "image" not in name:
                # Extract the Type of the Layer
                layer_type = name.split('.')[-1]
                
                # Add the Layer Type to the Set of Unique Layers
                unique_layers.add(name)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)
    
def print_named_parameters(model, use_print_data=False):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model

def preprocess_function_train(datapoint):
    prompt_column = 'question'
    response_column = 'answer'
    image_column = 'image'
    model_inputs = {
        "input_ids": None,
        "labels": None,
        "image_tensors": None
    }
    if datapoint[prompt_column] and datapoint[response_column]:
        query, answer = datapoint[prompt_column], datapoint[response_column]
        
        prompt = tokenizer.build_prompt(query,)

        a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                    max_length=128)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                    max_length=128)

        input_ids = a_ids
        labels = b_ids + [tokenizer.eos_token_id]

        model_inputs["input_ids"] =input_ids
        model_inputs["labels"] =labels
        image_tensor=feature_extractor(datapoint[image_column], return_tensors='pt')["pixel_values"].squeeze()
        model_inputs['image_tensors'] =image_tensor
        
    return model_inputs

def data_collator(batch):
    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels")) + 1
                     for i in range(len(batch))]
    len_max_batch = min(200, max(len_max_batch))
    batch_input_ids = []
    batch_labels = []
    batch_image_tensors = []
    for ba in batch:
        x, y, image_tensor = ba.get("input_ids"), ba.get("labels") , ba.get('image_tensors')
        len_padding = len_max_batch - len(x) - len(y)
        if tokenizer.padding_side and tokenizer.padding_side == "left":
            labels = [-100] * len_padding + [-100] * len(x) + y
            input_ids = [ID_PAD] * (len_padding) + x + y
        else:
            labels = [-100] * len(x) + y + [-100] * len_padding
            input_ids = x + y + [ID_PAD] * (len_padding)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        image_tensor = torch.tensor(image_tensor)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
        batch_image_tensors.append(image_tensor)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    batch_image_tensors = torch.stack(batch_image_tensors)
    input_dict = {
                "input_ids": batch_input_ids,
                "labels": batch_labels,
                "image_tensors":batch_image_tensors,
                }
    return input_dict
    
if __name__ == "__main__":
    tokenizer = ChatGLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Allow batched inference
    model = ChatGLMForConditionalGeneration.from_pretrained(PATH_MODEL_PRETRAIN)
    model.transformer.init_image_model(PATH_MODEL_IMAGE)
    feature_extractor = BeitImageProcessor.from_pretrained(PATH_MODEL_IMAGE)
    ## set target module to tune
    TARGET_MODULES = find_target_modules(model, tuning_module = TUNING_MODULE)
    model.config.use_cache = USE_CACHE
    config = LoraConfig(target_modules=TARGET_MODULES,
                        lora_dropout=LORA_DROPOUT,
                        lora_alpha=LORA_ALPHA,
                        task_type="CAUSAL_LM",
                        bias="none",
                        r=LORA_R,
                        )
    model = get_peft_model(model, config)
    model = model.cuda()
    model.print_trainable_parameters()
    
    for name , parameters in model.named_parameters():
        if "align" in name:
            parameters.requires_grad = True
    
    data = datasets.load_dataset('flaviagiammarino/vqa-rad')
    train_data = data["train"].map(preprocess_function_train).shuffle()
    
    trainer = CustomTrainer(
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #                     tokenizer, pad_to_multiple_of=8,
        #                     return_tensors="pt", padding=True
        #                 ),
        data_collator=data_collator,
        train_dataset=train_data,
        model=model,
        args=transformers.TrainingArguments(
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            max_grad_norm=0.5,
            logging_steps=20,
            warmup_steps=16,  # 618
            warmup_ratio=0.01,
            # warmup_steps=16,
            evaluation_strategy="no",
            lr_scheduler_type="cosine", #'constant',  # "cosine",
            logging_first_step=False,
            # evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            # eval_steps=SAVE_STEPS if VAL_SET_SIZE > 0 else None,
            save_strategy="steps",
            save_total_limit=32,
            save_steps=SAVE_STEPS,
            # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            # ddp_find_unused_parameters=None,
            gradient_checkpointing=True,
            # group_by_length=True,  # group together samples of roughly the same length in training
            output_dir=MODEL_SAVE_DIR,
            remove_unused_columns=False,
            optim="adamw_torch",  # "adamw_hf",
            report_to=[],  # ["tensorboard"],  # [], ["wandb"]
            fp16=True,
        )
    )
    
    wandb.init(project= "image–chatglm2")
    flag_checkpoint = False
    trainer.train(resume_from_checkpoint=flag_checkpoint)
    ## need to save both lora ab and prefix align layer.
    trainer.save_model()
    old_state_dict = model.state_dict
    
    
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    
    print_named_parameters(model, use_print_data=True)  # 查看LoRA层权重是不是为NAN溢出