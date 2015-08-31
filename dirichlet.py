# Dirichlet Classifier (Fast Counting)
# -----------------------------------
# A more packaged and shaped version of dirichlet counting that is possible for use in language model structure. 

import numpy as np
import numpy.random as npr

def unique_array(a):
  b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
  # This little hack is nice but it is not sorted since we replaced everything with void. We should sort it before it messes things up later.
  _, idx, inv = np.unique(b, return_index=True, return_inverse=True)
  key = np.argsort(idx)
  orig = np.arange(0, key.shape[0])
  d = dict(zip(key, orig))
  newinv = np.array([d[i] for i in inv])
  return idx[key], newinv

# Dirichlet is special in that in cannot return probabilities for 
# something that is has not seen before. 
def filler_proba(v):
  return np.ones(v) * 1 / float(v)

class dirichlet(object):
  def __init__(self, numV):
    self.num_vocab = numV
    self.keys = []
    self.values = []

  # inputs are the prediction_tuples
  # inputs to be a matrix, outputs to be a matrix and indexes to be a vector.
  def fit(self, inputs, outputs, indexes):
    inputs, outputs, indexes = np.array(inputs), np.array(outputs), np.array(indexes)
    num_vocab = self.num_vocab # Local var to reduce accesses
    uniq_inputs, uniq_key, uniq_inv = np.unique(inputs, return_index=True, return_inverse=True)
    uniq_indexes = indexes[uniq_key]
    # Another local variable - store number of uniques
    uniq_num = uniq_inputs.shape[0]
    # Create an array for each vocab class for each input class.
    counts = np.zeros((uniq_num, num_vocab))
    # Loop through all the inputs, outputs, indexes (non-unique) and add counts as appropriate.
    for i in range(inputs.shape[0]):
      counts[uniq_inv[i]][outputs[i]] += 1
    # Create a matrix of base alphas, add counts, and apply dirichlet
    alphas = np.ones((uniq_num, num_vocab)) / float(num_vocab)
    alphas += counts
    thetas = np.array([npr.dirichlet(a) for a in alphas])
    # Go ahead and just incorporate the results into the object
    self.keys = uniq_inputs # This is slow.
    self.values = thetas
  
  def predict_proba(self, inputs):
    num_v   = self.num_vocab
    d = dict(zip(self.keys, self.values))
    default = filler_proba(num_v)
    outputs = np.array([d[i] if i in d else default for i in inputs])
    return outputs

