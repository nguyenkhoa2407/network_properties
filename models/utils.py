import torch

def attach_relu_activation_hook(model):
  # hook fn
  def relu_activation_hook(layer_name, result_storage):
    def hook(_model, _inputs, outputs):
      result_storage[layer_name] = outputs > 0
    return hook
  
  handles = []
  activation_storage = {}
  for name, module in list(model.named_modules()):
    nested_modules = list(module.named_modules())[1:]
    if len(nested_modules) > 0: continue # only process the layers, ignore all other containers
    if isinstance(module, torch.nn.ReLU):
      handle = module.register_forward_hook(relu_activation_hook(name, activation_storage))
      handles.append(handle)  

  return handles, activation_storage
  

def attach_layer_output_hook(model):
  # hook fn
  def layer_output_hook(layer_name, result_storage):
    def hook(_model, _inputs, outputs):
      result_storage[layer_name] = outputs.detach()
    return hook
  
  handles = []
  output_storage = {}
  for name, module in list(model.named_modules()):
    nested_modules = list(module.named_modules())[1:]
    if len(nested_modules) > 0: continue # only process the layers, ignore all other containers
    handle = module.register_forward_hook(layer_output_hook(name, output_storage))
    handles.append(handle)  
  return handles, output_storage


def turn_bool_activation_to_str(activation_signature):
  for layer_name, activations in activation_signature.items():
    activation_signature[layer_name] = [
      ["ON" if val else "OFF" for val in activation]
      for activation in activations
    ]
  return activation_signature


def turn_bool_activation_to_int(activation_signature, to_list=False):
  for layer_name, activations in activation_signature.items():
    activation_signature[layer_name] = torch.where(activations, 1, 0)
    if to_list: 
      activation_signature[layer_name] = activation_signature[layer_name].tolist()
  return activation_signature
  
  
def get_layers_info(model):
  result = []

  for name, module in list(model.named_modules()):
    nested_modules = list(module.named_modules())[1:]
    # only process the layers, ignore all other containers
    if len(nested_modules) > 0: continue

    layer_info = {}
    # add more blocks here to handle different types of layers
    if isinstance(module, torch.nn.ReLU):
      # We'll use the preceding layers to get the info needed for ReLU layer
      prev_layer_info = result[-1]
      layer_info = {
        "name": name,
        "in_features": prev_layer_info["out_features"], 
        "out_features": prev_layer_info["out_features"], 
        "layer": module,
      }

    elif isinstance(module, torch.nn.Linear):
      layer_info = { 
        "name": name,
        "in_features": module.in_features, 
        "out_features": module.out_features, 
        "layer": module,
      }
      # if the first layer to be processed, then add information about the model's input layer
      # using the first layer's info
      if len(result) == 0:
        input_layer_info = {
          "name": "model_input",
          "in_features": layer_info["in_features"], 
          "out_features": layer_info["in_features"], 
          "layer": None,
        }
        result.append(input_layer_info)
        
    else: continue
      
    result.append(layer_info)
  return result