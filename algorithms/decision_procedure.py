import sys
import torch

sys.path.append('/Users/khoanguyen-cp/gmu/Marabou')

from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions
from models.utils import get_layers_info

class MarabouCoreDP():
  def __init__(self, lower_bound=-100, upper_bound=100):
    self.upper_bound = upper_bound
    self.lower_bound = lower_bound

  # 1. network_activation: a dictionary of layer_name -> layer_activation
  # 2. model: the model to be verified
  # 3. input_ranges: list of pairs for each input variable, i.e. [min, max]
  # 4. specification: provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
  #                   Each element in the list is a term in a disjunction for the specification.
  def solve(self, network_activation, model, input_ranges, specification):
    inputQuery = MarabouCore.InputQuery()
    layers_info = get_layers_info(model)

    for index, layer_info in enumerate(layers_info):
      layer = layer_info['layer']
      layer_name = layer_info['name']

      if layer_name == "model_input":
        inputQuery = self.__set_input_boundary(layer_info, inputQuery, input_ranges)

      elif layer_name == "final_output" or index == len(layers_info) - 1:
        inputQuery = self.__process_output_layer(layer_info, inputQuery, specification)
      
      elif isinstance(layer, torch.nn.ReLU):
        layer_activation = network_activation.get(layer_name)
        inputQuery = self.__process_relu_hidden_layer(layer_info, layer_activation, inputQuery)

      elif isinstance(layer, torch.nn.Linear):
        inputQuery = self.__process_linear_hidden_layer(layer_info, inputQuery)

    ## Run Marabou to solve the query
    options = createOptions(verbosity=0, snc=True)
    result = list(MarabouCore.solve(inputQuery, options, ""))
    result.append(inputQuery)
    return result
  

  ### HELPERS FOR INPUT LAYER ###

  def __set_input_boundary(self, layer_info, inputQuery, input_ranges):
    input_features = list(range(0, layer_info["in_features"]))
    inputQuery.setNumberOfVariables(len(input_features))
    # print(f"{layer_info['name']} has variables {input_features}")

    for var in input_features:
      var_min, var_max = input_ranges[var]
      inputQuery.setLowerBound(var, var_min)
      inputQuery.setUpperBound(var, var_max)
    return inputQuery
  

  ### HELPERS FOR OUTPUT LAYER ###

  def __process_output_layer(self, layer_info, inputQuery, specification):
    num_neurons = layer_info["out_features"]
    num_of_vars = inputQuery.getNumberOfVariables()
    layer_vars = list(range(num_of_vars, num_of_vars + num_neurons))
    # print(f"{type(layer_info['layer']).__name__} layer {layer_info['name']} has variables {layer_vars}")

    inputQuery.setNumberOfVariables(num_of_vars + num_neurons)
    inputQuery = self.__set_boundary_for_unconstrained_linear_vars(layer_vars, inputQuery)
    inputQuery = self.__set_linear_constraints(layer_vars, layer_info, inputQuery)
    inputQuery = self.__set_specification_constraints(layer_vars, specification, inputQuery)
    return inputQuery
  

  # specification_disjunction: 
  # Provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
  # Each element in the list is a term in a disjunction for the specification.
  def __set_specification_constraints(self, layer_vars, specification_disjunction, inputQuery):
    disjunction = []
    for mat, rhs in specification_disjunction:
      for mat_row, coefficients in enumerate(mat):
        rhs_val = rhs[mat_row]
        equation_type = MarabouCore.Equation.EquationType(2) # type 2 is Less than or equal (LE inequality)
        equation = MarabouCore.Equation(equation_type)

        for idx, var in enumerate(layer_vars):
          equation.addAddend(coefficients[idx], var)
        equation.setScalar(rhs_val)
        disjunction.append([equation])
    
    MarabouCore.addDisjunctionConstraint(inputQuery, disjunction)
    return inputQuery


  ### HELPERS FOR RELU LAYERS ###

  def __process_relu_hidden_layer(self, layer_info, layer_activation, inputQuery):
    num_neurons = layer_info["out_features"]
    num_of_vars = inputQuery.getNumberOfVariables()
    layer_vars = list(range(num_of_vars, num_of_vars + num_neurons))
    # print(f"{type(layer_info['layer']).__name__} layer {layer_info['name']} has variables {layer_vars}")

    if layer_activation == None: # unconstrained relu layer
      layer_activation = ["--" for _ in layer_vars]

    inputQuery.setNumberOfVariables(num_of_vars + num_neurons)
    inputQuery = self.__set_boundary_for_relu_vars(layer_vars, layer_activation, inputQuery)
    inputQuery = self.__set_relu_constraints(layer_vars, inputQuery)
    return inputQuery
  

  def __set_boundary_for_relu_vars(self, vars, layer_activation, inputQuery):
    for idx, var in enumerate(vars):
      if layer_activation[idx] == "ON": # var > 0
        inputQuery.setLowerBound(var, 1e-6)
        inputQuery.setUpperBound(var, self.upper_bound)

      elif layer_activation[idx] == "OFF":
        inputQuery.setLowerBound(var, 0)
        inputQuery.setUpperBound(var, 0)
        
      else: # neuron is unconstrained, can be on or off
        inputQuery.setLowerBound(var, 0)
        inputQuery.setUpperBound(var, self.upper_bound)
        
    return inputQuery
  

  def __set_relu_constraints(self, layer_vars, inputQuery):
    # each relu step is accompanied by a preceding layer of the same size
    prev_layer_vars = [var - len(layer_vars) for var in layer_vars]
    # print(f":::RELU CONSTRAINTS::: layer_vars: {layer_vars} - prev_layers_vars: {prev_layer_vars}\n")
    for idx, relu_var in enumerate(layer_vars):
      corresponding_prev_var = prev_layer_vars[idx]
      MarabouCore.addReluConstraint(inputQuery, corresponding_prev_var, relu_var)
    return inputQuery
  

  ### HELPERS FOR LINEAR LAYERS ###

  def __process_linear_hidden_layer(self, layer_info, inputQuery):
    num_neurons = layer_info["out_features"]
    num_of_vars = inputQuery.getNumberOfVariables()
    layer_vars = list(range(num_of_vars, num_of_vars + num_neurons))
    # print(f"{type(layer_info['layer']).__name__} layer {layer_info['name']} has variables {layer_vars}")

    inputQuery.setNumberOfVariables(num_of_vars + num_neurons)
    inputQuery = self.__set_boundary_for_unconstrained_linear_vars(layer_vars, inputQuery)
    inputQuery = self.__set_linear_constraints(layer_vars, layer_info, inputQuery)
    return inputQuery
  

  def __set_boundary_for_unconstrained_linear_vars(self, vars, inputQuery):
    for var in vars:
      inputQuery.setLowerBound(var, self.lower_bound)
      inputQuery.setUpperBound(var, self.upper_bound)
    return inputQuery
  

  def __set_linear_constraints(self, layer_vars, layer_info, inputQuery):
    # in dense layers, each neuron is connected to every neuron in the preceding layer
    # we can check the current layer's in_features to see the size of the preceding layer
    prev_layer_size = layer_info["in_features"]
    prev_layer_start_var = layer_vars[0] - prev_layer_size
    prev_layer_vars = list(range(prev_layer_start_var, prev_layer_start_var + prev_layer_size))
    # print(f":::LINEAR CONSTRAINTS::: layer_vars: {layer_vars} - prev_layers_vars: {prev_layer_vars}\n")

    # Ex: x2 = w0*x0 + w1*x1 + b
    # <=> w0*x0 + w1*x1 - x2 = -b
    layer = layer_info["layer"]
    for neuron_idx, var in enumerate(layer_vars):
      coefficients = layer.weight[neuron_idx]
      equation = MarabouCore.Equation()
      equation.addAddend(-1, var)
      for prev_idx, prev_var in enumerate(prev_layer_vars):
        equation.addAddend(coefficients[prev_idx], prev_var)
        
      scalar = 0.0 if layer.bias == None else (-layer.bias[neuron_idx])
      equation.setScalar(scalar)
      inputQuery.addEquation(equation)

    return inputQuery
  