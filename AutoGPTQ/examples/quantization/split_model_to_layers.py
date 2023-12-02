import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os

import torch
from transformers import AutoModelForCausalLM
import os

name = '2'
original_layer_weight = name[12:] + '.pt'


def save_layer(layer, layer_name, save_dir):
    """ Save the individual layer to disk """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(layer.state_dict(), os.path.join(save_dir, f"{layer_name}.pt"))

def load_layer(model, layer_name, save_dir):
    """ Load the individual layer from disk """
    layer_state_dict = torch.load(os.path.join(save_dir, f"{layer_name}.pt"))
    getattr(model, layer_name).load_state_dict(layer_state_dict)

def should_process_layer(name):
    """ Check if the layer should be processed based on its name """
    keywords = ['k_proj', 'out_proj', 'q_proj', 'v_proj', 'fc1', 'fc2']
    return any(keyword in name for keyword in keywords)

def modify_and_save_model(model, save_dir):
    """ Modify specific layers of the model and save them """
    for name, module in model.named_modules():
        if should_process_layer(name):
            print(f"Processing and saving layer: {name}")
            save_layer(module, name, save_dir)
        else:
            print(f"Skipping layer: {name}")

def load_modified_model(model_class, save_dir):
    """ Load the modified model from saved layers """
    model = model_class.from_pretrained(None)  # Initialize an empty model
    for name, _ in model.named_modules():
        if should_process_layer(name):
            load_layer(model, name, save_dir)
    return model

# Example Usage
model_name = "facebook/opt-13b"  # Replace with your model
save_dir = "/work/LAS/wzhang-lab/mingl/code/QMeZO/AutoGPTQ/examples/quantization/opt-13b-layers"

# Load the original model
config = AutoConfig.from_pretrained(model_name)
 # Auto device loading
torch_dtype = torch.float16
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map='auto',
    torch_dtype=torch_dtype,
)
print('begin to save')
# Modify and save parts of the model
modify_and_save_model(original_model.model, save_dir)

# Load the modified model
# modified_model = load_modified_model(AutoModelForCausalLM, save_dir)

