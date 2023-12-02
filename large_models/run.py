import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
from trainer import OurTrainer
import random

DEBUG = False
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP
    load_autogptq_model = True
    quantized_model_dir:str = '/work/LAS/wzhang-lab/mingl/code/QMeZO/AutoGPTQ/examples/quantization/opt-1.3b-2bit-128g'

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 

    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag
    save_model_mezo_part: bool = False


    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def clip_w(qweight, mezo, n_bit):
    """
    Clip the weights of mezo in place to ensure that 0 <= qweight + mezo.weight < 2 ** n_bit, without affecting gradients.

    :param qweight: The original quantized weight (a PyTorch tensor)
    :param mezo: The linear layer whose weights are to be clipped (a PyTorch nn.Module or nn.Parameter)
    :param n_bit: The number of bits for the quantization
    """
    # Define the minimum and maximum values based on n_bit
    min_val = 0
    max_val = (2 ** n_bit) - 1

    # Calculate the allowed range for mezo weights
    mezo_min = min_val - qweight
    mezo_max = max_val - qweight

    # Clip mezo weights to the allowed range without affecting the gradient
    with torch.no_grad():
        mezo.weight.data.reshape(qweight.shape).clamp_(mezo_min, mezo_max)

def quantize(tensor, bits):
    # Calculate the scale factor based on the number of bits
    # For signed integers, we use 2^(bits-1) - 1 to get the maximum value
    max_val = 2**(bits - 1) - 1
    scale = tensor.abs().max() / max_val

    # Quantize the tensor
    quantized_tensor = torch.round(tensor / scale) * scale
    return quantized_tensor

import torch

def custom_quantize(tensor, n_bits):
    # Determine the range for n-bit representation
    range_max = 2 ** n_bits - 1

    # Normalize tensor to 0 to range_max
    min_val, max_val = tensor.min(), tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val) * range_max

    # Quantize to n-bit values
    quantized_tensor = torch.round(normalized_tensor).int()
    return quantized_tensor, min_val, max_val

def custom_dequantize(quantized_tensor, min_val, max_val, n_bits):
    # Dequantize back to float
    range_max = 2 ** n_bits - 1
    dequantized_tensor = quantized_tensor.float() / range_max * (max_val - min_val) + min_val
    return dequantized_tensor

import os
import torch
from torch import Tensor
import math
import random
from torch import nn
import torch.nn.functional as F
from scipy.stats import norm
from torch import optim


class NFQuantizer:
    def __init__(self, num_bits=2, device='cuda', method='normal', block_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        self.method = method
        self.block_size = block_size
        if self.method == 'normal':
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif self.method == 'uniform':
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        else:
            raise NotImplementedError("Other quantization methods not supported yet.")

    @staticmethod
    def create_uniform_map(symmetric=False, num_bits=4):
        if symmetric:
            print("symmetric uniform quantization")
            negative = torch.linspace(-1, 0, 2 ** (num_bits - 1))
            positive = torch.linspace(0, 1, 2 ** (num_bits - 1))
            table = torch.cat([negative, positive[1:]])
        else:
            print("asymmetric uniform quantization")
            table = torch.linspace(-1, 1, 2 ** num_bits)
        return table

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        variations = 2 ** num_bits

        if symmetric:
            print("symmetric NormalFloat")
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            print("asymmetric NormalFloat")
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            # print(torch.linspace(offset, 0.5, 9)[:-1])
            # print(v1)
            v2 = [0]
            # v2 = [0]*(256-15) ## we have 15 non-zero values in this data type
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            # print(torch.linspace(offset, 0.5, 8)[:-1])
            # print(v3)
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        # print(values)
        return values
        # assert values.

    def quantize_tensor(self, weight):
        max_abs = torch.abs(weight).max()
        weight_normed = weight / max_abs

        weight_normed_expanded = weight_normed.unsqueeze(-1)

        # Reshape L to have the same number of dimensions as X_expanded
        L_reshaped = torch.tensor(self.norm_lookup_table).reshape(1, -1)

        # Calculate the absolute difference between X_expanded and L_reshaped
        abs_diff = torch.abs(weight_normed_expanded - L_reshaped)

        # Find the index of the minimum absolute difference for each element
        qweight = torch.argmin(abs_diff, dim=-1)
        # print(min_index)
        return qweight, max_abs

    def dequantize_tensor(self, qweight, max_abs):
        qweight_flatten = qweight.flatten()

        weight_normed = self.norm_lookup_table[qweight_flatten]
        weight = weight_normed * max_abs

        weight = weight.reshape(qweight.shape)

        return weight

    def quantize_block(self, weight):
        assert len(weight.shape) == 2 and weight.shape[0] * weight.shape[1] % self.block_size == 0
        M, N = weight.shape
        device = weight.device

        # Quantization
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B
        if self.method == 'normal':
            weight_max = weight_block.abs().max(dim=-1)[0]  # (L, 1)
        elif self.method == 'uniform':
            weight_max = weight_block.mean(dim=-1) + 2.5 * weight_block.std(dim=-1)
        else:
            raise NotImplementedError("Method not supported yet.")
        weight_max = weight_max.unsqueeze(-1)
        weight_divabs = weight_block / weight_max  # (L, B)
        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        # Pack multiple k-bit into uint8
        qweight = qweight.reshape(-1, 8 // self.num_bits)
        qweight_pack = torch.zeros((M * N // 8 * self.num_bits, 1), dtype=torch.uint8, device=device)

        # data format example:
        # [1, 0, 3, 2] or [01, 00, 11, 10]  -> [10110001], LIFO
        for i in range(8 // self.num_bits):
            qweight[:, i] = qweight[:, i] << i * self.num_bits
            qweight_pack[:, 0] |= qweight[:, i]

        return qweight_pack, weight_max, weight.shape

    def dequantize_block(self, qweight, weight_max, weight_shape):
        # unpack weight
        device = qweight.device
        weight = torch.zeros((qweight.shape[0], 8 // self.num_bits), dtype=torch.float32, device=device)
        for i in range(8 // self.num_bits):
            lookup_table_idx = qweight.to(torch.long) % 2 ** self.num_bits  # get the most right 2 bits
            lookup_table_idx = lookup_table_idx.to(torch.int)
            weight[:, i] = self.norm_lookup_table[lookup_table_idx].squeeze()
            qweight = qweight >> self.num_bits  # right shift 2 bits of the original data

        weight_block = weight.reshape(-1, self.block_size)
        weight = weight_block * weight_max
        weight = weight.reshape(weight_shape)

        return weight

def custom_forward(self, x):
        if not hasattr(self, 'wf'):
            # is performed by unpacking the weights and using torch.matmul
            if self.bits in [2, 4, 8]:
                self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
            elif self.bits == 3:
                self.wf = torch.tensor(
                    [
                        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                        [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                        [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                    ],
                    dtype=torch.int32
                ).reshape(1, 3, 12)

        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        if self.autogptq_cuda_available is True and (
            self.kernel_switch_threshold is False or x.shape[0] < self.kernel_switch_threshold
        ):
            out = torch.zeros(x.shape[0], out_shape[-1], dtype=torch.float, device=x.device)
            if self.use_cuda_fp16:
                x = x.half()
                if self.bits == 2:
                    self.autogptq_cuda.vecquant2matmul_faster_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size, self.half_indim)
                elif self.bits == 3:
                    self.autogptq_cuda.vecquant3matmul_faster_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size, self.half_indim)
                elif self.bits == 4:
                    self.autogptq_cuda.vecquant4matmul_faster_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size, self.half_indim)

                else:
                    raise NotImplementedError("Only 2,3,4 bits are supported.")
            else:
                x = x.float()
                if self.bits == 2:
                    self.autogptq_cuda.vecquant2matmul_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size)
                elif self.bits == 3:
                    self.autogptq_cuda.vecquant3matmul_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size)
                elif self.bits == 4:
                    self.autogptq_cuda.vecquant4matmul_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size)
                elif self.bits == 8:
                    self.autogptq_cuda.vecquant8matmul_old(x, self.qweight, out, self.scales.float(), self.qzeros, self.group_size)
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        else:
            if self.wf.device != self.qzeros.device:
               self.wf = self.wf.to(self.qzeros.device)
                
            if self.bits in [2,4,8]:
               zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits), self.wf.unsqueeze(0)).to(torch.int16 if self.bits == 8 else torch.int8)
               torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)
                   
               zeros = zeros + 1
               zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
   
               scales = self.scales
               scales = scales.reshape(-1, 1, scales.shape[-1])
                
               weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1), self.wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8)
               torch.bitwise_and(weight,(2 ** self.bits) - 1, out=weight)
               weight = weight.reshape(-1, self.group_size, weight.shape[2])
            elif self.bits == 3:
               zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1]//3, 3, 1).expand(-1, -1, -1, 12)
               zeros = (zeros >> self.wf.unsqueeze(0))
               zeros[:,:,0,10] = (zeros[:,:,0,10]&0x3) | ((zeros[:,:,1,0] << 2)&0x4)
               zeros[:,:,1,11] = (zeros[:,:,1,11]&0x1) | ((zeros[:,:,2,0] << 1)&0x6)
               zeros = zeros & 0x7
               zeros = torch.cat([zeros[:,:,0,:11], zeros[:,:,1,1:12], zeros[:,:,2,1:11]], dim=2)
                
               zeros = zeros + 1
               zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2]) 
               
               scales = self.scales
               scales = scales.reshape(-1, 1, scales.shape[-1])
                
               weight = self.qweight.reshape(self.qweight.shape[0]//3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
               weight = (weight >> self.wf.unsqueeze(-1))&0x7
               weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
               weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
               weight = weight & 0x7
               weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
               weight = weight.reshape(-1, self.group_size, weight.shape[2])
            else:
               raise NotImplementedError("Only 2,3,4,8 bits are supported.")
            weight = (scales * (weight - zeros))
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

            # intialise mezo part as original weight - quantized weight
            if self.original_layer_weight_dir:
                layer_state_dict = torch.load(self.original_layer_weight_dir)
                self.original_layer_weight_dir = None
                with torch.no_grad():
                    self.mezo_part.weight.data = layer_state_dict['weight'].data - weight.data.reshape(self.mezo_part.weight.shape)
                del layer_state_dict

            if hasattr(self, 'mezo_part'):
                weight += self.mezo_part.weight.reshape(weight.shape)
            elif hasattr(self, 'lora_A'):
                # weight += (self.lora_B @ self.lora_A).reshape(weight.shape)
                weight += (self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            out = torch.matmul(x.half(), weight)
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        if hasattr(self, 'mezo_part'):    
            out += self.mezo_part.bias

        return out

name_to_mezo_part = {}
def add_mezo_parts(model):
    original_weight_dir = "/work/LAS/wzhang-lab/mingl/code/QMeZO/AutoGPTQ/examples/quantization/opt-13b-layers"
    for name, module in model.named_modules():
        if 'mezo_part' in name:
            if 'weight' in name:
                name_to_mezo_part[name] = 1
            continue
        if 'k_proj' in name or 'out_proj' in name or 'q_proj' in name or 'v_proj' in name or 'fc1' in name or 'fc2' in name:
            logger.info(f'Inject mezo to name:{name} type:{type(module).__name__}')
            mezo_part = torch.nn.Linear(in_features=module.infeatures, out_features=module.outfeatures, bias=True)
            # initialse mezo part as zeros
            torch.nn.init.zeros_(mezo_part.weight)
            torch.nn.init.zeros_(mezo_part.bias)
            mezo_part.original_layer_weight_dir = original_weight_dir + name + '.pt'
            # use NFQuantizer
            # mezo_part.quantizer = NFQuantizer(num_bits=2, method='normal', device=model.device, block_size=64)
            # mezo_part.weight_size = torch.Size([module.outfeatures, module.infeatures])
            # mezo_part.weight_type = model.dtype
            mezo_part.to(device=model.device, dtype=model.dtype)
            mezo_part.weight.requires_grad = True
            mezo_part.bias.requires_grad = True
            module.mezo_part = mezo_part
            module.forward = custom_forward.__get__(module)
            module.use_cuda_fp16 = True
            module.autogptq_cuda_available = False
        

def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module

def reset_parameters(self):
    if hasattr(self, 'lora_A'):
        # initialize A the same way as the default for nn.Linear and B to zero
        if DEBUG:
            logging.info(f'Before:{self.lora_A}')
            logging.info(f'Before:{self.lora_B}')
        # nn.init.normal_(self.lora_A)
        mean = 0.0  # Mean of the normal distribution
        std = 0.01   # Standard deviation of the normal distribution
        # Initialize lora_A with normal distribution
        self.lora_A.data.normal_(mean, std)
        nn.init.zeros_(self.lora_B)
        if DEBUG:
            logging.info(f'After:{self.lora_A}')
            logging.info(f'After:{self.lora_B}')


def add_mezo_lora_parts(model, r, alpha, float16):
    for name, module in model.named_modules():
        if 'lora' in name:
            continue
        # if 'k_proj' in name or 'out_proj' in name or 'q_proj' in name or 'v_proj' in name or 'fc1' in name or 'fc2' in name:
        if 'q_proj' in name or 'v_proj' in name:
            logger.info(f'Inject mezo lora to name:{name} type:{type(module).__name__}')
            device, dtype = module.weight.device, torch.float16 if float16 else torch.float32
            if r > 0:
                module.lora_alpha = alpha
                module.r = r
                # module.lora_A = nn.Parameter(torch.zeros((r, module.in_features), device=device, dtype=dtype))
                # module.lora_B = nn.Parameter(torch.zeros((module.out_features, r), device=device, dtype=dtype))
                module.lora_A = nn.Parameter(module.bias.new_zeros((r, module.in_features)))
                module.lora_B = nn.Parameter(module.bias.new_zeros((module.out_features, r)))
                module.scaling = module.lora_alpha / module.r
                reset_parameters(module)
            module.forward = custom_forward.__get__(module)
            module.use_cuda_fp16 = True
            module.autogptq_cuda_available = False

    # Freeze non-LoRA parameters
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False

def add_mezo_lora_parts_old(model, r, alpha, float16):
    if model.config.model_type == "opt":
        attention_name = "attn"
    elif model.config.model_type == "roberta":
        attention_name = "attention"
    else:
        raise NotImplementedError

    # Insert LoRA
    for key, _ in model.named_modules():
        if key[-len(attention_name):] == attention_name:
            logger.info(f"Inject mezo lora to: {key}")
            _, _, attn = find_module(model, key)

            if model.config.model_type == "opt":
                original_q = attn.q_proj
                device, dtype = original_q.weight.device, torch.float16 if float16 else torch.float32
                if r > 0:
                    original_q.lora_alpha = alpha
                    original_q.r = r
                    # original_q.lora_A = nn.Parameter(torch.zeros((r, original_q.in_features), device=device, dtype=dtype))
                    # original_q.lora_B = nn.Parameter(torch.zeros((original_q.out_features, r), device=device, dtype=dtype))
                    original_q.lora_A = nn.Parameter(original_q.bias.new_zeros((r, original_q.in_features)))
                    original_q.lora_B = nn.Parameter(original_q.bias.new_zeros((original_q.out_features, r)))
                    original_q.scaling = original_q.lora_alpha / original_q.r
                    reset_parameters(original_q)
                original_q.forward = custom_forward.__get__(original_q)
                original_q.use_cuda_fp16 = True
                original_q.autogptq_cuda_available = False
                
                original_v = attn.v_proj
                if r > 0:
                    original_v.lora_alpha = alpha
                    original_v.r = r
                    original_v.lora_A = nn.Parameter(torch.zeros((r, original_v.in_features), device=device, dtype=dtype))
                    original_v.lora_B = nn.Parameter(torch.zeros((original_v.out_features, r), device=device, dtype=dtype))
                    original_v.scaling = original_v.lora_alpha / original_v.r
                    reset_parameters(original_v)
                original_v.forward = custom_forward.__get__(original_v)
                original_v.use_cuda_fp16 = True
                original_v.autogptq_cuda_available = False
            else:
                raise NotImplementedError
    
    # Freeze non-LoRA parameters
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False

class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                # Head tuning
                from ht_opt import OPTForCausalLM
                model = OPTForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            elif self.args.quantized_model_dir:
                quantized_model_dir = '/work/LAS/wzhang-lab/mingl/code/QMeZO/AutoGPTQ/examples/quantization/opt-13b-2bit-32g'
                from auto_gptq import AutoGPTQForCausalLM
                model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)
                model.eval()
                if (self.args.train_set_seed is not None or self.args.num_train_sets is not None) and not self.args.lora:
                    add_mezo_parts(model)
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype,
                    max_memory={i: f'{free_in_GB-5}GB' for i in range(torch.cuda.device_count())},
                    load_in_8bit=self.args.load_int8,
                )
                
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            if not self.args.quantized_model_dir:
                from lora import LoRA
                LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)
            else:
                add_mezo_lora_parts(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer


    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids = input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        if self.args.label_names is None:
            self.args.label_names = ['labels']
            
        trainer = OurTrainer(
            model=self.model, 
            args=self.args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )
        trainer.name_to_mezo_part = name_to_mezo_part
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=last_checkpoint) 
        torch.cuda.empty_cache()

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        if self.args.save_model_mezo_part:
            logger.warn("Save model mezo part..")
            # Assuming 'model' is your PyTorch model
            model_state_dict =trainer.model.state_dict()
            mezo_layers = {name: param for name, param in model_state_dict.items() if 'mezo' in name}
            torch.save(mezo_layers, f'{self.args.output_dir}/mezo_layers.pth')
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        # if self.args.only_train_option and not self.args.non_diff:
        #     if type(self.model) == FSDP:
        #         logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
        #         self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
        #     else:
        #         self.model.forward = self.model.original_forward


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    args = parse_args()

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)

if __name__ == "__main__": 
    main()