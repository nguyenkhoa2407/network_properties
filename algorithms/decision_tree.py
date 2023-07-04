from sklearn import tree
import graphviz 

class DecisionTree():
  def __init__(self, df, X_col="X", Y_col="Y"):
    self.X = list(df[X_col])
    self.Y = list(df[Y_col])
    self.clf = tree.DecisionTreeClassifier()

    
  def get_potential_layer_properties(self):
    self.__fit()

    children_left = self.clf.tree_.children_left
    children_right = self.clf.tree_.children_right
    feature = self.clf.tree_.feature
    impurity = self.clf.tree_.impurity
    value = self.clf.tree_.value
    classes = self.clf.classes_
    
    # Traverse through tree to get leaves and the paths that lead to these leaves
    leaves = []
    # each stack element follows the format (node_id: int, activation_pattern: list)
    # each element in activation_pattern is itself another list [neuron_id, activation_status]
    stack = [(0, [])]

    # DFS
    while len(stack) > 0: 
      node_id, activation_pattern = stack.pop()

      # If the left and right child of a node is not the same we have a split node
      is_split_node = children_left[node_id] != children_right[node_id]

      # If a split node, append left and right children and depth to `stack` so we can loop through them
      if is_split_node:
        neuron_id = feature[node_id]
        stack.append((children_left[node_id], activation_pattern + [[neuron_id, "OFF"]]))
        stack.append((children_right[node_id], activation_pattern + [[neuron_id, "ON"]]))
        
      else: # if leaf node, see if it's pure AND satisfies postcondition
        class_index = value[node_id].argmax()
        class_label = classes[class_index]
        support = int(max(value[node_id].flatten()))
        processed_activation_pattern = self.__process_leaf_activation_pattern(activation_pattern)

        sat_postcond = (class_label == 1)
        is_pure = (impurity[node_id] == 0)
        
        leaf_data = {
          "node_id": node_id, 
          "sat_postcond": sat_postcond,
          "support": support,
          "activation_pattern": processed_activation_pattern
        }
        if is_pure and sat_postcond: leaves.append(leaf_data)
    
    # sort by support, in descending order
    sorted_leaves = sorted(leaves, key=lambda x: -x['support'])
    return sorted_leaves
  

  def __process_leaf_activation_pattern(self, unprocessed_pattern):
    # default pattern where every neuron is unconstrained
    result = [ "--" for _ in range(len(self.X[0])) ] 
    for neuron_id, activation_status in unprocessed_pattern:
      result[neuron_id] = activation_status
    return result
  

  def __fit(self):
    self.clf = self.clf.fit(self.X, self.Y)
  
  
  def visualize_tree(self):
    feature_names = [ f"n{i}" for i in range(len(self.X[0])) ]
    dot_data = tree.export_graphviz(
      self.clf, 
      out_file=None, 
      feature_names= feature_names,  
      class_names=["false", "true"],  
      filled=True, 
      rounded=True,  
      special_characters=True
    )
    return graphviz.Source(dot_data)  