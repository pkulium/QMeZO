from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_id = "facebook/opt-6.7b"

quantization_config = GPTQConfig(
     bits=2,
     group_size=1024,
     dataset="c4",
     desc_act=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map='auto')
quant_model.save_pretrained("opt-6.7b-gptq")
