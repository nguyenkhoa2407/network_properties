

import copy
import torch
from algorithms.decision_procedure import MarabouCoreDP
from models.utils import attach_relu_activation_hook, turn_bool_activation_to_str

class IterativeRelaxation():
  def __init__(self):
    self.dp = MarabouCoreDP()

  def call(self, model, input_data, input_ranges, specification):
    # attach hooks to model to get activation signature of X
    _act_handles, activation_signature = attach_relu_activation_hook(model)

    # evaluate model with X to get activation signature of X
    X = torch.tensor(input_data, dtype=torch.float)
    _logits = model(X)

    activation_signature = self.__process_activation_signature(activation_signature)
    status, _, _, _ = self.dp.solve(activation_signature, model, input_ranges, specification)
    if status == "sat":
      return [activation_signature, specification]
    
    layer_names = list(activation_signature.keys())
    max_unconstrained_layer_idx = len(layer_names) - 1
    unconstrained_layer_idx = max_unconstrained_layer_idx

    while unconstrained_layer_idx >= 0: 
      # print(f"unconstrained_layer_id: {unconstrained_layer_idx}")
      
      # unconstrain all neurons in the layer
      layer_name = layer_names[unconstrained_layer_idx]
      print(f"unconstrained_layer: {layer_name}")
      original_activation = activation_signature[layer_name]
      activation_signature[layer_name] = ["--" for val in original_activation]
      print(activation_signature)
      
      status, _, _, _ = self.dp.solve(activation_signature, model, input_ranges, specification)
      
      if status == "sat": # critical layer found
        print(f"Critical layer found: {unconstrained_layer_idx}")
        
        crit_layer_idx = unconstrained_layer_idx
        crit_layer_name = layer_names[crit_layer_idx]
        # add back activations from critical layer
        activation_signature[crit_layer_name] = copy.deepcopy(original_activation)
        crit_layer_activation = activation_signature[crit_layer_name]
        
        # iteratively unconstrain neurons to see if they are needed
        for neuron_idx, _val in enumerate(activation_signature[crit_layer_name]):
          crit_layer_activation[neuron_idx] = "--"
          print(f"--- unconstraining neuron {neuron_idx} in critical layer")
          # print(activation_signature)
          status, _, _, _ = self.dp.solve(activation_signature, model, input_ranges, specification)
          
          if status == "sat": # neuron needed, must remain constrained
            print(f"--- neuron needed")
            crit_layer_activation[neuron_idx] = original_activation[neuron_idx]
        
        return [activation_signature]
      
      else: 
        unconstrained_layer_idx -= 1


  def __process_activation_signature(self, activation_signature):
    activation_signature = turn_bool_activation_to_str(activation_signature)
    for layer_name, activations in activation_signature.items():
      activation_signature[layer_name] = activations[0]
    return activation_signature
