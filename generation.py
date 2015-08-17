import numpy as np, sys
import numpy.random as npr
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
# ------------------------
sys.path.append('../')
import lmixture, hmm
import cross_validation as cv2
import z_scored_words as zs

# We are assuming that the images are in the 
# ../data folder.
def fetch_paths(num, path='../data/'):
  ids = range(1, num+1)
  names = np.array([str(i)+'.png' for i in ids])
  paths = np.array([fullpath + i for i in names])
  return paths

# Sample from an image but ensure that the samples
# are consecutively ordered.
def consecutive_sample(arr, num):
  sample = npr.randint(0, arr.shape[0]-num)
  return arr[sample:sample+num]

# This makes the image pixels with lower resolution,
# so we can group NxN groups before sampling.
def blockify(block_size, image, num_sample):
  image_cols, image_rows = image.shape[1], image.shape[0]
  col_iterator, row_iterator = image_cols / block_size, image_rows / block_size
  # Iterate through rows FIRST, then columns.
  block_mat = np.array([image[j*block_size:(j+1)*block_size, i*block_size:(i+1)*block_size] for i in range(col_iterator) for j in range(row_iterator)])
  block_seq = np.array([np.sum(np.sum(block_mat[i], axis=0), axis=0) / block_size**2 for i in range(block_mat.shape[0])]) 
  # Sample an appropriate amount from each image.
  return consecutive_sample(block_seq, num_sample)

# Instead of doing things like blocks, we want to 
# organize the styles by rows, columns, and diagonals. 
# This makes sense since image patterns aren't just horizontal.
def sample_image(image, num2sample=500, style='row'):
  if not style in ['row', 'column', 'down-diag', 'up-diag']:
      style = 'row'
  # Grab the dimensions of the image. 
  row_count, col_count = image.shape[0], image.shape[1]
  feat_count = image.shape[2]
  if style == 'row':
      # Flatten the image and then sample X points.
      image = image.reshape((row_count*col_count, feat_count))
      samples = consecutive_sample(image, num2sample)
  elif style == 'column':
      # transpose and then flatten.
      image = image.transpose((1, 0, 2))
      image = image.reshape((row_count*col_count, feat_count))
      samples = consecutive_sample(image, num2sample)
  elif style == 'down-diag':
      # This is a bit harder. Make indexes so that we can track
      # position as we are manipulating things.
      indexes = np.array(range(row_count * col_count))
      indexes = indexes.reshape((row_count, col_count))
      # Now to get the diagonal matrix, we can use np.diag. 
      diags = np.array([np.diag(indexes, k=k) for k in range(-1*row_count+1, col_count)])
      # Reshape to flat image, apply, concat, and finish. 
      image = image.reshape((row_count*col_count, feat_count))
      image = np.concatenate([image[d] for d in diags])
      samples = consecutive_sample(image, num2sample)
  elif style == 'up-diag':
      # Same thing as down-diag but need to transpose first.
      indexes = np.array(range(image.shape[0] * image.shape[1]))
      indexes = indexes.reshape((image.shape[0], image.shape[1]))
      indexes = np.fliplr(indexes)
      diags = np.array([np.diag(indexes, k=k) for k in range(-1*row_count+1, col_count)])
      image = image.reshape((row_count*col_count, feat_count))
      image = np.concatenate([image[d] for d in diags])
      samples = consecutive_sample(image, num2sample)
  return samples

# The random forest may return a slice of the probabilities,
# and so we want to re-join them.
def reverse_bag(classOrd, probSeq):
  fullSeq = np.array([[1e-25 for i in V] for j in probSeq])
  fullSeq[:, classOrd] += probSeq
  fullSeq /= np.sum(fullSeq, axis=1).reshape((fullSeq.shape[0], 1))
  return fullSeq

# Sample using a flip().
def sample(v, r):
  rolling = 0
  for i in xrange(len(v)):
      rolling += v[i]
      if r <= rolling:
          return i

# Scripts to generate the first seed for 
# image generation.
def gen_seed():
  seed1 = npr.randint(0, 255, 3) # Seed 1 is always a temp
  seed2 = npr.randint(0, 255, 3)
  seed  = np.array([seed1, seed2])
  return seed

# This performs the actual testing. It will return 
# a probability upon a sample.
def gen_value(X, y, I, forestMat, T, priorProbs):
  test_inputs, test_targets, test_models, test_indexes, test_raw = preprocess_fun(X, y, I)
  tupleMat, dataMat, placeMat = setup_fun(test_inputs, test_indexes, test_models)
  resample, priorProbs, saveProbs, logsum = backward_fun(dataMat, placeMat, test_models, forestMat, T, priorProbs)
  test_models = lmixture.update(test_indexes, test_models, resample)
  curr_model = test_models[0][1]
  
  classifiers = forestMat[curr_model] # Get the classifier
  probabilities = []
  for dim in range(F):
      probabilities.append(reverse_bag(classifiers[dim].classes_, classifiers[dim].predict_proba(test_inputs[0][1]))[0])
  probabilities = np.array(probabilities)
  return probabilities

# Little hack to use the MLM library.
def pad_sample(sample):
  seed = npr.randint(0, 255, 3)
  return np.array([seed, sample])

# For viewing purposes, we might want to maximize each row!
def expand(arr, size):
  return np.array([[arr for i in range(size)] for j in range(size)])

def magnify(image, factor):
  m = np.array([np.concatenate([expand(image[row][col], factor) for col in range(image.shape[1])]) for row in range(image.shape[0])])
  return m.transpose((1, 0, 2))

# Run this on DEFAULT. Some settings should be changed.
if __name__ == '__main__':

  paths = fetch_paths(30)
  datum = np.array([np.asarray(Image.open(i)) for i in paths])
  # Pull out different types of samples from each image.
  organized = []
  for image in datum:
    organized.append(sample_image(image, 500, 'row'))
    organized.append(sample_image(image, 500, 'column'))
    organized.append(sample_image(image, 500, 'down-diag'))
    organized.append(sample_image(image, 500, 'up-diag'))
  organized = np.array(organized)
  # Format these into X, y, and I 
  I = np.concatenate([np.ones(organized[i].shape[0])*i for i in range(len(organized))])
  X = np.concatenate(organized)
  # Just do some dummy thing since we don't need outcomes. 
  # We don't need outcomes here, since we are sampling for 
  # generation. Let's just do something dumb.
  y = np.zeros(X.shape[0]) 
  # Let's get rid of an extra dim.
  dimensions = X.shape[1]
  X = X[:, 0:dimensions-1]
  # Set up parameters for mixture of language models.
  N = 4 
  F = dimensions - 1
  V = np.array(range(0, 256))
  R = 'random-forest'
  # Use the language models.
  preprocess_fun, setup_fun, forward_fun, backward_fun = lmixture.build_lmixture(N, F, V, R)
  # Generate a transition matrix for HMM
  T = hmm.generate_T_mat(N, 0, 0.8)
  # Run 25 iterations:
  train_inputs, train_targets, train_models, train_indexes, train_raw = preprocess_fun(X, y, I, clusterInit='random')
  tupleMat, dataMat, placeMat = setup_fun(train_inputs, train_indexes, train_models)
  forestMat = forward_fun(tupleMat)
  resample, priorProbs, saveProbs, logsum = backward_fun(dataMat, placeMat, train_models, forestMat, T)
  train_models = lmixture.update(train_indexes, train_models, resample)
  currIter = 2 
  while (currIter < 25):
    tupleMat, dataMat, placeMat = setup_fun(train_inputs, train_indexes, train_models)
    forestMat = forward_fun(tupleMat)
    resample, priorProbs, saveProbs, logsum = backward_fun(dataMat, placeMat, train_models, forestMat, T)
    train_models = lmixture.update(train_indexes, train_models, resample)
    print("Iteration: %d." % currIter)
    currIter += 1
  # Do the image generation now. 
  returned_values = [] # <-- save it here.
  seed = gen_seed()
  # Hardcoded ytest and Itest --> they don't matter for this.
  Xtest, ytest, Itest = np.array(seed), np.zeros(2), np.ones(2)
  probs = gen_value(Xtest, ytest, Itest, forestMat, T, priorProbs)
  new_sample = np.array([sample(pr, npr.rand()) for pr in probs])
  returned_values.append(new_sample)
  # Generate 250x250 for now.
  gen_size = 250
  gen_area = gen_size**2
  for itr in range(gen_area):
      if itr % 1000 == 0:
          print(itr)
      cur_sample = pad_sample(new_sample)
      Xtest, ytest, Itest = np.array(cur_sample), np.zeros(2), np.ones(2)
      probs = gen_value(Xtest, ytest, Itest, forestMat, T, priorProbs)
      new_sample = np.array([sample(pr, npr.rand()) for pr in probs])
      returned_values.append(new_sample)
  # Arrayify this for easy viewing
  returned_values = np.array(returned_values)
  # Transform the returned_values into an image
  returned_values = returned_values[:gen_area]
  # Add the old 255 dimension back in.
  returned = np.insert(returned_values, 3, 255, axis=1)
  resized = np.uint8(returned.reshape((gen_size, gen_size, 4)))
  im = Image.fromarray(resized)
  # Show the image 
  fig_size = 8
  plt.figure(figsize=(fig_size, fig_size))
  plt.imshow(im)


