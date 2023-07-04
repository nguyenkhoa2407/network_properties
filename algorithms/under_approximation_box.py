import pulp 
import torch
import numpy as np

from models.utils import get_layers_info

class UnderApproximationBox():
  def solve(self, input_property, attr_min, attr_max, model):
    layers_info = get_layers_info(model)

    # Prepare inputs and objective function
    num_of_inputs = len(attr_min)
    pulp_problem, pulp_inputs = self.__prepare_inputs_and_objective_fn(num_of_inputs, attr_min, attr_max)

    # Constraints for neurons in input_property
    for idx, layer_info in enumerate(layers_info):
      layer = layer_info['layer']
      layer_name = layer_info['name']

      if layer_name == "model_input": continue
      elif layer_name == "final_output": continue
      
      elif isinstance(layer, torch.nn.Linear):
        prev_layer_info = layers_info[idx - 1]
        self.__set_linear_constraints(layer_info, prev_layer_info, input_property)

      elif isinstance(layer, torch.nn.ReLU):
        prev_layer_info = layers_info[idx - 1]
        default_activation = ["--" for _ in range(layer_info['out_features'])]
        activation = input_property.get(layer_name, default_activation)
        pulp_problem = self.__set_relu_constraints(layer_info, prev_layer_info, activation, pulp_problem, pulp_inputs)

    result = pulp_problem.solve()
    for v in pulp_problem.variables():
      print(v.name, "=", v.varValue)

    return pulp_problem, result
  

  def __set_linear_constraints(self, layer_info, prev_layer_info, input_property):
    current_layer = layer_info['layer']
    prev_layer = prev_layer_info['layer']

    if prev_layer_info['name'] == 'model_input':
      layer_info['weight_wrt_input'] = current_layer.weight

    elif isinstance(prev_layer, torch.nn.ReLU):
      default_activation = ["--" for _ in range(prev_layer_info['out_features'])]
      prev_activations = input_property.get(prev_layer_info['name'], default_activation)
      layer_info['weight_wrt_input'] = torch.tensor([])

      for neuron in range(layer_info['out_features']):
        # a tensor of size (1 x prev_layer_info['out_features'])
        weight_wrt_to_prev_layer = [
          current_layer.weight[neuron][idx].item() if activation == "ON" else 0.0
          for idx, activation in enumerate(prev_activations)
        ]
        weight_wrt_to_prev_layer = torch.tensor([weight_wrt_to_prev_layer])

        # a tensor of size (prev_layer_info['out_features'] x num_of_inputs)
        prev_layer_weight_wrt_to_input = prev_layer_info['weight_wrt_input']

        # final result is tensor of size (1 x num_of_inputs)
        weight_wrt_input = torch.mm(weight_wrt_to_prev_layer, prev_layer_weight_wrt_to_input)
        layer_info['weight_wrt_input'] = torch.cat((layer_info['weight_wrt_input'], weight_wrt_input))


  def __set_relu_constraints(self, layer_info, prev_layer_info, activation, pulp_problem, pulp_inputs):
    prev_layer = prev_layer_info['layer']
    weight_wrt_input = prev_layer_info['weight_wrt_input']

    # for relu layer, its "weight" is the same as the prev layer
    layer_info['weight_wrt_input'] = weight_wrt_input

    for neuron in range(layer_info["in_features"]):
      coefficients = weight_wrt_input[neuron]
      neuron_activation = activation[neuron]

      expressions = []
      for idx, (d_lo, d_hi) in enumerate(pulp_inputs):
        if coefficients[idx] >= 0: # d_hi
          expressions.append(coefficients[idx].item() * d_hi)
        else: 
          expressions.append(coefficients[idx].item() * d_lo)

      bias = 0.0 if prev_layer.bias == None else (- prev_layer.bias[neuron])
      if neuron_activation == "ON":
        pulp_problem += (-(pulp.lpSum(expressions)) <= bias)
      elif neuron_activation == "OFF":
        pulp_problem += (pulp.lpSum(expressions) <= -bias)

    return pulp_problem

  
  def __prepare_inputs_and_objective_fn(self, num_of_inputs, attr_min, attr_max):
    inputs = []
    # set up input variables
    for i in range(0, num_of_inputs):
      d_hi = pulp.LpVariable(f"d_hi_{i}", lowBound=attr_min[i], upBound=attr_max[i])
      d_lo = pulp.LpVariable(f"d_lo_{i}", lowBound=attr_min[i], upBound=attr_max[i])
      inputs.append((d_lo, d_hi))

    # objective function
    problem = pulp.LpProblem("UnderApproximationBox", pulp.LpMaximize)
    problem += pulp.lpSum([d_hi - d_lo for d_lo, d_hi in inputs])
    return problem, inputs
