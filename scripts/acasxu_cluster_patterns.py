import torch
import copy
import json
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
sys.path.append('..')

from algorithms.decision_procedure import MarabouCoreDP
from algorithms.clustering import dbscan_clustering, get_samples_from_clusters, get_cluster_ranges, get_cluster_min_max_features
from models.utils import attach_relu_activation_hook, turn_bool_activation_to_int
from models.acasxu_1_1 import Acasxu1_1
from datasets.clusterinACAS_0_shrt import read_inputs_from_file, create_df, prediction_class_specification


def __find_cluster_decision_patterns(clusters, samples, dnn):
  _, activation_signature = attach_relu_activation_hook(dnn)
  activation_counts_template = __initialize_activation_counts(dnn, [[0.62, 0.1, 0.2, 0.47, -0.48]])  
  _ = dnn(torch.tensor(samples, dtype=torch.float32))
  activation_signature = turn_bool_activation_to_int(activation_signature, to_list=True)
  
  num_samples_per_cluster = int(len(samples)/len(clusters))
  candidates = {}
  
  for cluster in clusters: 
    activation_counts = copy.deepcopy(activation_counts_template)
    
    for layer, counts in activation_counts.items():
      start_id = cluster * num_samples_per_cluster
      end_id = start_id + num_samples_per_cluster
      cluster_activations = activation_signature[layer][start_id:end_id]
      
      for activation in cluster_activations:
        counts += np.array(activation)
      
    cluster_candidate = {}
    for layer, neuron_act_counts in activation_counts.items():
      new_activation = [
        "ON" if count == num_samples_per_cluster else ("OFF" if count == 0 else "--")
        for count in neuron_act_counts
      ]
      cluster_candidate[layer] = new_activation
    candidates[cluster] = cluster_candidate
    
  return candidates


def __verify_cluster_decision_patterns(patterns_by_cluster, cluster_ranges, dnn_model, specifications):
  dp = MarabouCoreDP()
  examined_candidates = set()
  accepted_candidates = []
  
  for cluster, candidate_pattern in patterns_by_cluster.items(): 
    print(f"Verifying pattern of cluster {cluster}: ")
    print(candidate_pattern)
    if json.dumps(candidate_pattern) in examined_candidates: 
      continue
      
    ranges = cluster_ranges[cluster]
    status, _, __, ___ = dp.solve(candidate_pattern, dnn_model, ranges, specifications)
    examined_candidates.add(json.dumps(candidate_pattern))
    
    if status == "unsat": 
      accepted_candidates.append((cluster, candidate_pattern))

  print(f"\nAccepted {len(accepted_candidates)} decision patterns.")
  return accepted_candidates


def __initialize_activation_counts(dnn, sample):
  _, activation_signature = attach_relu_activation_hook(dnn)  
  X = torch.tensor(sample, dtype=torch.float)
  _ = dnn(X)
  activation_signature = turn_bool_activation_to_int(activation_signature)
  for layer_name, activations in activation_signature.items():
    activation_signature[layer_name] = np.zeros_like(activations[0])
  return copy.deepcopy(activation_signature)


def __get_input_features(df, y_class):
  class_df = df[df['predicted_label'] == y_class]
  input_features = np.vstack(np.array(class_df['input']))
  return input_features


if __name__ == "__main__":
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

  # Get input features that have predicted class 0 and find clusters within the gathered features
  print(" ===== Getting input features for class 0 and finding clusters ===== ")
  X = __get_input_features(df, 0)
  dbscan_point_counts, dbscan_labels = dbscan_clustering(X, 0.142, 10)
  dbscan_clusters = dbscan_point_counts.keys()
  dbscan_cluster_ranges = get_cluster_ranges(dbscan_clusters, X, dbscan_labels)

  # Find candidate decision patterns for each cluster
  print(" ===== Find candidate decision patterns for each cluster ===== ")
  dbscan_num_samples = min(min(dbscan_point_counts.values()), 2000)
  dbscan_samples = get_samples_from_clusters(dbscan_num_samples, dbscan_clusters, X, dbscan_labels)
  dbscan_candidates = __find_cluster_decision_patterns(dbscan_clusters, dbscan_samples, dnn_model)

  # Verify decision patterns using Marabou
  print(" ===== Verifying decision patterns using Marabour ===== ")
  specification = prediction_class_specification(0)
  accepted_candidates = __verify_cluster_decision_patterns(dbscan_candidates, dbscan_cluster_ranges, dnn_model,specification)
  print("Accepted candidates: ")
  for accepted_candidate in accepted_candidates: 
    print(accepted_candidate)
    print('---------------------------------')