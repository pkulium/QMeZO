from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_id = "facebook/opt-13b"

quantization_config = GPTQConfig(
     bits=2,
     group_size=128,
     dataset="c4",
     desc_act=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map='auto', low_cpu_mem_usage=True)
# quant_model.save_pretrained("opt-13b-gptq", safe_serialization=True)
torch.save(quant_model.state_dict(), 'opt-13b-gptq')
