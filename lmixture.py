# Mixture of Language Models - Build by Mike Wu
# ---------------------------------------------
# Because everything is in list comprehension, things might be a little
# difficult to understand. Sorry about that! But because things were not
# sized uniformly, numpy can't help in many circumstances. 

# -- Uses Dirichlet Counting/Random Forest/Naive Bayes for prediction. 
# -- Uses Gibbs Sampling to calculate generative probabilities
# -- Uses Hidden Markov Models for smoothed resampling.
from __future__ import absolute_import
from __future__ import print_function
import sys, copy, itertools
# Need this for LM initialization
from sklearn.cluster import KMeans
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt
# Need this for randomized forests
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
# For sampling (backwards algorithms)
import hmm, dirichlet
sys.path.append('../')
import z_scored_words as zs
sys.path.append('../neural-nets/')

# ----------------------------------------------------------------------
class MixtureParams(object):
  def __init__(self):
    self.vocabulary   = []
    self.num_models   = 0
    self.num_features = 0
    self.resampling_method = 'random-forest'

  # Trivial attribute definitions.
  def set_vocabulary(self, array):
    self.vocabulary   = array

  def set_num_models(self, num):
    self.num_models   = num

  def set_num_features(self, num):
    self.num_features = num

  def set_resampling_method(self, string):
    self.resampling_method = 'dirichlet'
    if string in ['dirichlet', 'random-forest', 'naive-bayes']:
      self.resampling_method = string

  # Given a number of models, 4 different simple initializations.
  # [Random init, K-Means init, all zeroes init, all ones init]
  def get_clusters(self, data, init='random'):
    length = data.shape[0]
    if init == 'random': 
      # We may also want to randomly initialize it just to be able 
      # to avoid any local maxima's that k-means traps us in.
      return np.random.randint(self.num_models, size=length) 
    elif init == 'k-means':
      kmeans = KMeans(init='k-means++', n_clusters=self.num_models, n_init=5)
      kmeans.fit(data)
      return kmeans.predict(data) # Predict on itself to get clusters
    elif init == 'zeros':
      return np.zeros(length)
    elif init == 'ones':
      return np.ones(length)

# ---------------------------------------------------------------------------
# Useful fxns used in main def. This recombines an arbitrary number
# of arrays by some index I. 
def regroup_by_patient(I, A):
  patients = np.unique(I)
  return [A[I == p] for p in patients]

# Translates into discrete 11 "words" (integers).
def make_alphabet(A, N):
  X = zs.discretize(A)
  return zs.alphabetize(X, N)

# This class is meant to be used to add stochasticity 
# to the predicted probabilities by adding a small decaying
# negative number. 
class MixtureDecay(object):
  # These rates are tested and should not really be messed with
  def __init__(self, seed=1, rate=0.008, maxiter=100):
    self.seed    = seed # Default decay starting from 1
    self.rate    = rate # Default decay rate 
    self.maxiter = maxiter # Scaling factor
    self.curiter = 1 
    self.factor  = 1000 # This tells us when the settings reach 0
    self.set_factor()

  # You also have the option of changing these to
  # more appropriate values if need be. 
  def set_rate(self, num):
    self.rate = num

  def set_seed(self, num):
    self.seed = num
  
  def set_factor(self):
    i = 0
    while self.test(i, self.seed) > 0.001:
      i+=1
    self.factor = i

  def update_iter(self, num):
    self.curiter = num

  # If we want to add stochasticity, then we can do so with 
  # exponential decay. 
  def exp_decay(self, N):
    def decay(t): # We want to divide by the maxiter here so that we scale to 0 nicely.
      return N*np.exp(-1*self.rate*t*float(self.factor)/self.maxiter)
    return decay
  
  # Sole purpose is to find the point at which the exponential decay func
  # should stop to be near 0;
  def test(self, t, N):
    return N*np.exp(-1*self.rate*t)
  
  # Given a rate and a seed, generate a single random value.
  def generate(self):
    return self.exp_decay(self.seed)

  # Depending on the iteration and the struct, do something.
  def apply(self, struct):
    func = self.generate()
    dim1, dim2 = struct.shape[0], struct.shape[1]
    randGen = npr.rand(dim1, dim2)
    struct -= randGen * func(self.curiter)
    return struct

# ---------------------------------------------------------------------------
# Special definitions to prevent tuple zipping
# Manually do list zipping.
def list_(*args): return list(args)
def zip_(*args): return map(list_, *args)

# Special flatten fxn for my nested structs.
# Must be a list of lists.
def flatten(struct):
  return np.array(list(itertools.chain.from_iterable(struct)))

def greater(arr, num):
  return arr[arr > num]

def charify(A):
  B = A.astype('str')
  # Join them by - so that if we have vocab > 10, it's okay
  B = np.array([''.join(row) for row in B])
  return B

def split_and_pick(s,d):
  return int(s.split('-')[d])

# Useful function to calculate complete likelihoods across iters
# (Hidden Function)
def get_summed_likelihood(proba, model):
  return np.sum([p[m] for p,m in zip(proba, model)])

# ---------------------------------------------------------------------------
# Inputs are the training sets for X, y, and I
# where I is the index of patients
def build_lmixture(N, method='random-forest'):
  # Note: You run a cycle, you want to run the following commands:
  # setup, forward_pass, backward_pass, resample. To measure convergence, define some epsilon and look for changes in likelihood between iterations.
  # Preprocess should be run beforehand.
  params = MixtureParams()
  params.set_num_models(N)   
  params.set_resampling_method(method)
  # Generally, we use a 11 integer vocab: range(-5, 6)

  # In the case that we are passed normal vectors, this fxn
  # provides the opportunity to alphabetize it, regroup by
  # one axis, and to cast language models on top.
  def preprocess(X, y, I, clusterInit='random'):
    # Store an ID for each index
    ID = np.arange(len(I))
    # Calculate the language splits
    clusters = params.get_clusters(X, init=clusterInit)
    # Alphabetize the X array
    vlen = 4
    X2 = make_alphabet(X, vlen)
    # Update different parmam settings based on results
    params.set_num_features(X2.shape[1])
    params.set_vocabulary(range(0, vlen*2 + 1))
    # Abstract into the inputs we want.
    inputs   = regroup_by_patient(I, X2)
    targets  = regroup_by_patient(I, y)
    models   = regroup_by_patient(I, clusters)
    indexes  = regroup_by_patient(I, ID)
    raw      = regroup_by_patient(I, X)
    # indexes will be useful because in the 
    # forward_pass, we move to the unique space (unofficially)
    # In the backwards_pass, we want to back to
    # non-unique spaces. In that case, indexes will serve
    # as the uniqueness tracker.
    # ---------
    # A little hack for improving dirichlet speed. It converts everything to chars and concats them into strings for processing.
    if method == 'dirichlet':
      inputs = setup_dirichlet(inputs)
    return inputs, targets, models, indexes, raw

  # In order to make the code a bit more robust, we 
  # want to do an initial processing step before gibbs
  # to generate the model tuples. 
  # ---------------------------------------------------
  # I apologize for the readibility. Because sizes are not
  # uniform, I can't use numpy and list comprehension was the
  # best I could do. Sorry :(. There is another file called
  # "readable" that would be better!
  def setup(inputs, indexes, models):
    # Find indexes of each of the models and store the index (but don't take the ones with 0th entry (since there is no previous point before 0).
    selectArr = np.array([[greater(np.arange(len(p))[p == lm], 0) for p in models] for lm in xrange(N)])
    # Now for each of the x_tn, we have to make a shit ton of tuples. 
    # for each d in x_tn, make the tuple (x_dtn, [x_1,t-1,n ,  ..., x_D,t-1,n])
    # x_tn for current LM
    monsterMat = np.array([[[zip_(inputs[i][data-1], inputs[i][data]), zip_(indexes[i][data-1], indexes[i][data])] for i, data in enumerate(selection)] for selection in selectArr])
    dataMat = np.array([np.array(list(itertools.chain(*u))) for u in monsterMat[:, :, 0]])
    placeMat = np.array([np.array(list(itertools.chain(*u))) for u in monsterMat[:, :, 1]]) 
    return dataMat, placeMat

# ---------------------------------------------------------------------------
  # Special Dirichlet Setup. In order to have things go by faster, we need this
  # And we want to plug this into the setup instead of train_inputs directly.
  # Basically we actually do need to hash things. (but only for dirichlet).
  def setup_dirichlet(train_inputs):
    return [charify(struct) for struct in train_inputs]

  # Dirichlet Forward Pass
  # ----------------------
  # Given the processed tuples, we want to update the language model projects
  # given the hidden parameters theta.

  def train_dirichlet(inputs, outputs, indexes, num_vocab):
    diri = dirichlet.dirichlet(numV=num_vocab)
    diri.fit(inputs, outputs, indexes)
    return diri

  def forward_dirichlet(data, places):
    F, V = params.num_features, params.vocabulary
    # F, V = 1, [0, 1]
    diriMat = [[train_dirichlet(data[lm][:, 0], [x[d] for x in data[lm][:, 1]], places[lm][:, 1], len(V)) for d in xrange(F)] for lm in xrange(N)]
    return diriMat

  def backward_dirichlet(data, places, models, diriMat, TMat, priorMat=None, decay=None, temper=[1,1]):

    def get_log_prior():
      # Add counts to the beta distribution from new layer
      betas = np.ones(N) / float(N)
      model_type, model_count = np.unique(np.concatenate(models), return_counts=True)
      for i in range(len(model_type)):
          betas[model_type[i]] += model_count[i]
      parameters = np.log(npr.dirichlet(betas))
      return parameters

    F, V = params.num_features, params.vocabulary
    # F, V = 1, [0, 1]
    # Let's calculate the priors. 
    prior = get_log_prior() if priorMat is None else priorMat
    # Setup, we want to go through all the keys together
    total_data, total_places = np.concatenate(data), np.concatenate(places)
    inputs, outputs = total_data[:, 0], total_data[:, 1]
    currPlaces, nextPlaces = total_places[:, 0], total_places[:, 1]
    # Calculations for p(y_t | x_t)
    probId, probArr = nextPlaces, np.zeros((nextPlaces.shape[0], N))
    # Loop through each of trained diri's and pull out probabilities
    for lm in range(N):
      for d in range(F):
        diri = diriMat[lm][d]
        # Predict probabilities on its own inputs
        logproba = np.log(diri.predict_proba(inputs))
        logproba = np.array([p[o] for p,o in zip(logproba, [x[d] for x in outputs])])
        probArr[:, lm] += logproba
    # Before continuing, I must sort it.
    sortKey = np.argsort(probId)
    probId  = probId[sortKey]
    probArr = probArr[sortKey]
    # If we have a decay object, use it (the iteration is updated outside)
    if decay is not None: decay.apply(probArr) # in-place changing
    # Save a copy of the probDict since we need to return it.
    saveArr = probArr + prior
    # Do a little prepping for the HMM run.
    flatten_models = np.concatenate(models)[probId]
    # Initialize a hidden Markov model to calculate smoothed posterior.
    hmm_forward, hmm_backward = hmm.build_hmm(N, TMat)
    forward_struct = hmm_forward(probArr, prior, temper)
    resampled, backward_struct = hmm_backward(forward_struct, flatten_models)
    # Calculate a total summed likelihood (finding the maximum theta likelihood throughout all lm's for each data vec).
    post_summed_log_likelihood = get_summed_likelihood(probArr, resampled)
    return resampled, probId, prior, saveArr, post_summed_log_likelihood    

# ---------------------------------------------------------------------------
  # Random Forest Forward Pass
  # --------------------------
  # An alternative to the forward-backward dirichlet. This uses Random Forests to calculate probabilities. The procedure for running these then becomes z --> randomize --> loop. 
  def train_forest(inputs, outputs, num_trees=10):
    forest = ExtraTreesClassifier(n_estimators=num_trees, random_state=42)
    forest.fit(inputs, outputs)
    return forest

  def forward_randomize(data, num_trees=10):
    # F = 1
    F = params.num_features
    forestMat = [[train_forest(data[lm][:, 0], data[lm][:, 1, d], num_trees) for d in xrange(F)] for lm in xrange(N)]
    return forestMat

  # Random Forest Backward Pass
  # ---------------------------
  # Decay is a tuned exponential decay fxn with randomness for stochasticity
  def backward_randomize(data, places, models, forestMat, TMat, priorMat=None, decay=None, temper=[1,1]):
    def get_log_prior():
      # Add counts to the beta distribution from new layer
      betas = np.ones(N) / float(N)
      model_type, model_count = np.unique(np.concatenate(models), return_counts=True)
      for i in range(len(model_type)):
          betas[model_type[i]] += model_count[i]
      parameters = np.log(npr.dirichlet(betas))
      return parameters

    def reverse_bag(classOrd, probSeq):
      fullSeq = np.ones((probSeq.shape[0], len(V))) * 1e-25
      fullSeq[:, classOrd] += probSeq
      fullSeq /= np.sum(fullSeq, axis=1).reshape((fullSeq.shape[0], 1))
      return fullSeq

    F, V = params.num_features, params.vocabulary
    # F, V = 1, [0, 1]
    # Let's calculate the priors. This is the same as in the dirichlet model.
    prior = get_log_prior() if priorMat is None else priorMat
    # Setup, we want to go through all the keys together
    total_data, total_places = np.concatenate(data), np.concatenate(places)
    inputs, outputs = total_data[:, 0], total_data[:, 1]
    currPlaces, nextPlaces = total_places[:, 0], total_places[:, 1]
    # Calculations for p(y_t | x_t)
    probId, probArr = nextPlaces, np.zeros((nextPlaces.shape[0], N))
    # Loop through the forests and calculate probabilities.
    for lm in xrange(N):
      for d in xrange(F):
        forest = forestMat[lm][d]
        logproba = np.log(reverse_bag(forest.classes_, forest.predict_proba(inputs)))
        probArr[:, lm] += np.array([p[o] for p,o in zip(logproba, outputs[:, d])])
    # Before continuing, I must sort it.
    sortKey = np.argsort(probId)
    probId  = probId[sortKey]
    probArr = probArr[sortKey]
    # If we have a decay object, use it (the iteration is updated outside)
    if decay is not None: decay.apply(probArr) # in-place changing
    # Save a copy of the probDict since we need to return it.
    saveArr = probArr + prior
    # Do a little prepping for the HMM run.
    flatten_models = np.concatenate(models)[probId]
    # Initialize a hidden Markov model to calculate smoothed posterior.
    hmm_forward, hmm_backward = hmm.build_hmm(N, TMat)
    forward_struct = hmm_forward(probArr, prior, temper)
    resampled, backward_struct = hmm_backward(forward_struct, flatten_models)
    # Calculate a total summed likelihood (finding the maximum theta likelihood throughout all lm's for each data vec).
    post_summed_log_likelihood = get_summed_likelihood(probArr, resampled)
    return resampled, probId, prior, saveArr, post_summed_log_likelihood

# ---------------------------------------------------------------------------

  # Naive Bayes Forward Pass
  # --------------------------
  # Another alternative to the Dirichlet model. This will make use of a naiveBayes predictor.
  def train_bayes(inputs, outputs, nbtype='gaussian'):
    if (nbtype == 'bernoulli'):
      bayes = BernoulliNB()
    elif (nbtype == 'multinomial'): 
      bayes = MultinomialNB()
    else:
      bayes = GaussianNB()
    bayes.fit(inputs, outputs)
    return bayes

  def forward_bayesify(data, nbtype='gaussian'):
    F = params.num_features
    # F = 1
    bayesMat = [[train_bayes(data[lm][:, 0], data[lm][:, 1, d], nbtype) for d in xrange(F)] for lm in xrange(N)]
    return bayesMat

  # Naive Bayes Backward Pass
  # ---------------------------
  def backward_bayesify(data, places, models, bayesMat, TMat, priorMat=None, decay=None, temper=[1,1]):
    def get_log_prior():
      # Add counts to the beta distribution from new layer
      betas = np.ones(N) / float(N)
      model_type, model_count = np.unique(np.concatenate(models), return_counts=True)
      for i in range(len(model_type)):
        betas[model_type[i]] += model_count[i]
      parameters = np.log(npr.dirichlet(betas))
      return parameters

    # Sometimes in naive bayes, we get this weird thing where it appends columns of all 0's. Let's remove those columns.
    def parse_columns(arr):
      return arr[:, np.array([i for i in xrange(arr.shape[1]) if (np.sum(arr[:, i] > 0) > 0)])]

    # Since we don't want to risk the probabilities to be 0, we should add something to it to help. It seems like naive bayes also reverse bags. We want to this for an entire sequence array at a time.
    def reverse_bag(classOrd, probSeq):
      # Call parse_columns in case there are extra appendages.
      if classOrd.shape[0] != probSeq.shape[1]:
        probSeq = parse_columns(probSeq)
      fullSeq = np.ones((probSeq.shape[0], len(V))) * 1e-25
      fullSeq[:, classOrd] += probSeq
      fullSeq /= np.sum(fullSeq, axis=1).reshape((fullSeq.shape[0], 1))
      return fullSeq

    F, V = params.num_features, params.vocabulary
    # F, V = 1, [0, 1]
    # Let's calculate the priors. This is the same as in the dirichlet model.
    prior = get_log_prior() if priorMat is None else priorMat
    # Setup, we want to go through all the keys together
    total_data, total_places = np.concatenate(data), np.concatenate(places)
    inputs, outputs = total_data[:, 0], total_data[:, 1]
    currPlaces, nextPlaces = total_places[:, 0], total_places[:, 1]
    # Calculations for p(y_t | x_t)
    probId, probArr = nextPlaces, np.zeros((nextPlaces.shape[0], N))
    # Loop through the forests and calculate probabilities.
    for lm in xrange(N):
      for d in xrange(F):
        bayes = bayesMat[lm][d]
        logproba = np.log(reverse_bag(bayes.classes_, bayes.predict_proba(inputs)))
        probArr[:, lm] += np.array([p[o] for p,o in zip(logproba, outputs[:, d])])
    # Before continuing, I must sort it.
    sortKey = np.argsort(probId)
    probId  = probId[sortKey]
    probArr = probArr[sortKey]
    # If we have a decay object, use it (the iteration is updated outside)
    if decay is not None: decay.apply(probArr) # in-place changing
    # Save a copy of the probDict since we need to return it.
    saveArr = probArr + prior
    # Do a little prepping for the HMM run.
    flatten_models = np.concatenate(models)[probId]
    # Initialize a hidden Markov model to calculate smoothed posterior.
    hmm_forward, hmm_backward = hmm.build_hmm(N, TMat)
    forward_struct = hmm_forward(probArr, prior, temper)
    resampled, backward_struct = hmm_backward(forward_struct, flatten_models)
    # Calculate a total summed likelihood (finding the maximum theta likelihood throughout all lm's for each data vec).
    post_summed_log_likelihood = get_summed_likelihood(probArr, resampled)
    return resampled, probId, prior, saveArr, post_summed_log_likelihood

  # All backwards functions merely return a array of probabilities, This function will take transform that array into the model_format using train_indexes! 
  def update(indexes, models, resample, resampleId):
    split_points = np.cumsum([i.shape[0] for i in indexes])
    all_indexes = np.concatenate(indexes)
    all_models = np.concatenate(models)
    # Find indexes of resampled in total
    fmask = np.in1d(all_indexes, resampleId)
    # Create a temporary dict and funcs for easy lookup 
    d = dict(zip(resampleId, resample))
    g = lambda x: d[x]
    lookup = map(g, all_indexes[fmask])
    # Do the actual updating.
    all_models[fmask] = lookup
    recombine = np.split(all_models, split_points)
    # Remove any extra empty ones (can be created by stupid splits)
    return np.delete(recombine, np.where([(u.size == 0) for u in recombine]))

  # Wrapper Forward Pass
  # ---------------------
  def forward(*args):
    if method == 'dirichlet':
      return forward_dirichlet(*args)
    elif method == 'random-forest':
      return forward_randomize(*args)
    elif method == 'naive-bayes':
      return forward_bayesify(*args)

  # Wrapper Backward Pass
  # ---------------------
  def backward(*args):
    if method == 'dirichlet':
      return backward_dirichlet(*args)
    elif method == 'random-forest':
      return backward_randomize(*args)
    elif method == 'naive-bayes':
      return backward_bayesify(*args)

  print("Successfully defined a Mixture of Language Models classifier with %d languages resampled using %s." % (N, method))
  return preprocess, setup, forward, backward, update

# ---------------------------------------------------------------------------

# Given the resulting probabilities, we want to convert those into organized feature vectors (+ extract corresponding labels) This will be used in both training and testing.
def vectorize(probId, probArr, inputs, targets, indexes, augment=True):
  # Flatten them so that it makes it easy for us to find things.
  flat_inputs = np.concatenate(inputs)
  flat_targets = np.concatenate(targets)
  flat_indexes = np.concatenate(indexes)
  # Find locations of each key of the probabilities in the total indexes
  lotsOfIdx = np.array([np.where(flat_indexes == i)[0][0] for i in probId])
  new_inputs = flat_inputs[lotsOfIdx]
  new_targets = flat_targets[lotsOfIdx]
  new_indexes = flat_indexes[lotsOfIdx]
  # Return some inputs (augmented or not).
  return_inputs = np.concatenate([probArr, new_inputs], axis=1) if augment else probArr
  return return_inputs, new_targets, new_indexes

