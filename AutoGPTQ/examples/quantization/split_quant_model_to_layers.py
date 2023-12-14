import os

from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np
import torch
import torch.nn as nn

bit, group = 4, 32
pretrained_model_dir = "facebook/opt-1.3b"
quantized_model_dir = f"opt-1.3b-{bit}bit-{group}g"

# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset, testenc

def custom_quant_forward(self, x):
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
        if self.name is not None:
            logging.info(f'save layer:{self.name}')
            torch.save(weight, f'{quantized_model_dir}/layers/{self.name}.pth')
            self.name = None

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
    for name, module in model.named_modules():
        if 'k_proj' in name or 'out_proj' in name or 'q_proj' in name or 'v_proj' in name or 'fc1' in name or 'fc2' in name:
            module.name = name
            logging.info(f'Modify layer:{name} type:{type(module).__name__}')
            module.forward = custom_quant_forward.__get__(module)
    

@torch.no_grad()
def opt_eval(model, testenc, dev, seqlen = 2048):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    traindataset,testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)

    quantize_config = BaseQuantizeConfig(
        bits=bit,  # quantize model to 4-bit
        group_size=group,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and group size only works on triton
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
    # with value under torch.LongTensor type.
    model.quantize(traindataset, use_triton=False)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    # save quantized model using ssafetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)

    # load quantized model, currently only support cpu or single gpu
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)
    add_mezo_parts(model)
    opt_eval(model.model, testenc, "cuda:0")

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
