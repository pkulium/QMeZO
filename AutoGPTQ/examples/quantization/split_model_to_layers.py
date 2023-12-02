import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os

def save_layer(layer, layer_name, save_dir):
    """ Save the individual layer to disk """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(layer.state_dict(), os.path.join(save_dir, f"{layer_name}.pt"))

def load_layer(model, layer_name, save_dir):
    """ Load the individual layer from disk """
    layer_state_dict = torch.load(os.path.join(save_dir, f"{layer_name}.pt"))
    getattr(model, layer_name).load_state_dict(layer_state_dict)

def modify_and_save_model(model, save_dir):
    """ Modify each layer of the model and save it """
    for name, module in model.named_children():
        print(f"Processing {name}")
        # Add your modification code here. For now, we'll just save the layer.
        save_layer(module, name, save_dir)

def load_modified_model(model_class, save_dir):
    """ Load the modified model from saved layers """
    model = model_class.from_pretrained(None)  # Initialize an empty model
    for name, _ in model.named_children():
        load_layer(model, name, save_dir)
    return model

# Example Usage
model_name = "facebook/opt-13b"  # Replace with your model
save_dir = "/work/LAS/wzhang-lab/mingl/code/QMeZO/AutoGPTQ/examples/quantization/opt-13b-layers"

# Load the original model
config = AutoConfig.from_pretrained(model_name)
original_model = AutoModelForCausalLM.from_pretrained(model_name)
print('begin to save')
# Modify and save parts of the model
modify_and_save_model(original_model, save_dir)

# Load the modified model
# modified_model = load_modified_model(AutoModelForCausalLM, save_dir)

