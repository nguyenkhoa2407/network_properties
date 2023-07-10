import pandas as pd
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')

def read_inputs_from_file(filepath='../datasets/clusterinACAS_0_shrt.csv'):
  num = 0 
  with open(filepath) as f:
    lines = f.readlines()
    # print(len(lines), "examples")
    features = np.empty([len(lines),5],dtype=float)
    labels = np.zeros(len(lines),dtype=int)

    for l in range(len(lines)):
      k = [float(stringIn) for stringIn in lines[l].split(',')] 
      if len(k) > 5:
        lab = int(k[5])
        labels[l+num] = lab
      for i in range(0,5):
        features[l+num][i] = k[i]
  return features, labels


def create_df(inputs, predicted_labels, true_labels):
  data = []
  for index, input_data in enumerate(inputs):
    data_point = { 
      "input": input_data, 
      "true_label": true_labels[index], 
      "predicted_label": predicted_labels[index].item(),
    }
    data.append(data_point)
  return pd.DataFrame(data)


def prediction_class_specification(y_class):
  classes_specifications = {
    0: [(np.array([[-1, 1, 0, 0, 0]]), np.array([0])),
        (np.array([[-1, 0, 1, 0, 0]]), np.array([0])),
        (np.array([[-1, 0, 0, 1, 0]]), np.array([0])),
        (np.array([[-1, 0, 0, 0, 1]]), np.array([0]))],
    
    1: [(np.array([[1, -1, 0, 0, 0]]), np.array([0])),
        (np.array([[0, -1, 1, 0, 0]]), np.array([0])),
        (np.array([[0, -1, 0, 1, 0]]), np.array([0])),
        (np.array([[0, -1, 0, 0, 1]]), np.array([0]))],
    
    2: [(np.array([[1, 0, -1, 0, 0]]), np.array([0])),
        (np.array([[0, 1, -1, 0, 0]]), np.array([0])),
        (np.array([[0, 0, -1, 1, 0]]), np.array([0])),
        (np.array([[0, 0, -1, 0, 1]]), np.array([0]))],
    
    3: [(np.array([[1, 0, 0, -1, 0]]), np.array([0])),
        (np.array([[0, 1, 0, -1, 0]]), np.array([0])),
        (np.array([[0, 0, 1, -1, 0]]), np.array([0])),
        (np.array([[0, 0, 0, -1, 1]]), np.array([0]))],
    
    4: [(np.array([[1, 0, 0, 0, -1]]), np.array([0])),
        (np.array([[0, 1, 0, 0, -1]]), np.array([0])),
        (np.array([[0, 0, 1, 0, -1]]), np.array([0])),
        (np.array([[0, 0, 0, 1, -1]]), np.array([0]))],
  }
  return classes_specifications[y_class]