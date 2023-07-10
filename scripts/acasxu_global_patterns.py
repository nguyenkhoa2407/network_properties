import torch
import copy
import json
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
sys.path.append('..')

from algorithms.decision_procedure import MarabouCoreDP
from models.utils import attach_relu_activation_hook, turn_bool_activation_to_int
from models.acasxu_1_1 import Acasxu1_1
from datasets.clusterinACAS_0_shrt import read_inputs_from_file, create_df, prediction_class_specification


def __find_global_decision_patterns(inputs, dnn):
  _, activation_signature = attach_relu_activation_hook(dnn)
  activation_counts = __initialize_activation_counts(dnn, [[0.62, 0.1, 0.2, 0.47, -0.48]])  

  _ = dnn(torch.tensor(inputs, dtype=torch.float32))
  inputs_activation_signature = turn_bool_activation_to_int(activation_signature, to_list=True)

  # calculate activation count of each neuron in the network
  for i in range(len(inputs)):
    for layer, activation_of_inputs in inputs_activation_signature.items():
      activation_counts[layer] += np.array(activation_of_inputs[i])

  global_decision_pattern = {}
  for layer, neuron_act_counts in activation_counts.items():
    global_layer_activation = [
      "ON" if count == len(inputs) else ("OFF" if count == 0 else "--")
      for count in neuron_act_counts
    ]
    global_decision_pattern[layer] = global_layer_activation
  
  return global_decision_pattern


def __get_input_features(df, y_class):
  class_df = df[df['predicted_label'] == y_class]
  input_features = np.vstack(np.array(class_df['input']))
  return input_features


def __initialize_activation_counts(dnn, sample):
  _, activation_signature = attach_relu_activation_hook(dnn)  
  X = torch.tensor(sample, dtype=torch.float)
  _ = dnn(X)
  activation_signature = turn_bool_activation_to_int(activation_signature)
  for layer_name, activations in activation_signature.items():
    activation_signature[layer_name] = np.zeros_like(activations[0])
  return copy.deepcopy(activation_signature)


def __find_input_ranges(df, n_features):
  ranges = []
  for i in range(n_features):
    min_val = df['input'].apply(lambda x: x[i]).min()
    max_val = df['input'].apply(lambda x: x[i]).max()
    ranges.append([min_val, max_val])
  return ranges


if __name__ == "__main__":
  y_class = 0 # TODO: change this to the class you want to verify, get val from command line params

  # load acasxu model and trained params
  print(" ===== Loading acasxu model and trained params ===== ")
  dnn_model = Acasxu1_1()
  dnn_model.load_state_dict(torch.load('../models/acasxu_1_1.pt'))

  # Load acasxu training dataset
  print(" ===== Loading acasxu training dataset ===== ")
  acas_train, acas_train_labels = read_inputs_from_file("../datasets/clusterinACAS_0_shrt.csv")
  outputs = dnn_model(torch.tensor(acas_train, dtype=torch.float32))
  predicted_labels = torch.argmin(outputs, dim=1)
  df = create_df(acas_train, predicted_labels, acas_train_labels)

  print(f" ===== Finding global pattern for class {y_class} ===== ")
  class_df = df[df['predicted_label'] == y_class]
  class_input_features = __get_input_features(class_df, y_class)
  global_decision_pattern = __find_global_decision_patterns(class_input_features, dnn_model)
  input_ranges = __find_input_ranges(class_df, 5)
  specification = prediction_class_specification(y_class)
  print(global_decision_pattern)

  print(" ===== Verifying global pattern ===== ")
  dp = MarabouCoreDP()
  res = dp.solve(global_decision_pattern, dnn_model, input_ranges, specification)
  print(res)

