
# Mixture of Language Models. 
# Build by Mike Wu

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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
# For sampling (backwards algorithms)
import hmm
import z_scored_words as zs

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
		self.resampling_method = 'random-forest'

	# Given a number of models, 4 different simple initializations.
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

# Useful fxns used in main def.
def regroup_by_patient(I, *args):
	patients = np.unique(I)
	data = [[] for struct in args]
	for p in patients:
		for i in range(len(args)):
			data[i].append(args[i][I == p])
	# Now we can return them as list
	# Casting np.arrays on this will make numpy angry
	return data

def make_alphabet(A):
	X = copy.deepcopy(A) # important
	indices = X.shape[1]
	# First make each row discrete
	for i in range(indices):
		X[:, i] = zs.discretize(X[:, i])
	return zs.alphabetize(X)

def make_image_alphabet(A):
	return A

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
		for key in struct.keys():
			for th in range(len(struct[key])):
				struct[key][th] -= npr.rand() * func(self.curiter)
   
# Special definitions to prevent tuple zipping
# Manually do list zipping.
def list_(*args): return list(args)
def zip_(*args): return map(list_, *args)

# Special flatten fxn for my nested structs.
# Must be a list of lists.
def flatten(struct):
	return np.array(list(itertools.chain.from_iterable(struct)))

# Hard code a nested array split
def specialSplit(i):
	return [arraySplit(j) for j in i]

def arraySplit(i): # convert (i) in to an array of numbers.
	return [ord(t) for t in i]

# Inputs are the training sets for X, y, and I
# where I is the index of patients
def build_lmixture(N, F, V, method='random-forest'):
	# Note: Yo run a cycle, you want to run the following commands:
	#       setup, forward_pass, backward_pass, resample.
	# 		To measure convergence, define some epsilon and look for 
	#       changes in likelihood between iterations.
	#	 	Preprocess should be run beforehand.

	params = MixtureParams()
	params.set_num_models(N) # Change me if you want
	params.set_num_features(F) # Change me if you want
	params.set_vocabulary(V) # Change me if you want
	params.set_resampling_method(method)
	 # This is for the char-->int conversion
	intMinV = min(params.vocabulary)
	# Generally, a good vocab is [chr(i) for i in range(97, 108)]

	# In the case that we are passed normal vectors, this fxn
	# provides the opportunity to alphabetize it, regroup by
	# one axis, and to cast language models on top.
	def preprocess(X, y, I, clusterInit='random'):
		# Store an ID for each index
		ID = np.array(range(len(I)))
		# Calculate the language splits
		clusters = params.get_clusters(X, init=clusterInit)
		# Regroup the structs by patient
		struct   = regroup_by_patient(I, X, y, clusters, ID)
		# Abstract into the inputs we want.
		inputs   = struct[0]
		targets  = struct[1]
		models   = struct[2]
		indexes  = struct[3]
		raw      = struct[0]
		# indexes will be useful because in the 
		# forward_pass, we move to the unique space (unofficially)
		# In the backwards_pass, we want to back to
		# non-unique spaces. In that case, indexes will serve
		# as the uniqueness tracker.
		return inputs, targets, models, indexes, raw

	# In order to make the code a bit more robust, we 
	# want to do an initial processing step before gibbs
	# to generate the model tuples. 
	def setup(inputs, indexes, models):
		tuplesMat = [] # Stores (x_t-1,n , x_dtn)
		dataMat   = [] # stores (x_t-1,n, x_t,n)
		placeMat  = [] # stores the indicies for (x_t-1,n, x_t,n)
		for lm in range(params.num_models):
			data_for_LM = []
			# Find indexes of each of the models and store the index
			for p in models:
				fraction = np.array(xrange(len(p)))[p == lm]
				data_for_LM.append(fraction[fraction > 0])

			tuplesVec = [] # x_tn for current LM
			# Now for each of the x_tn, we have to make a shit ton of tuples. 
			# for each d in x_tn, make the tuple (x_dtn, [x_1,t-1,n ,  ..., x_D,t-1,n])
			for dim in range(params.num_features):
				# These dim == 0 are because tuple's don't change by dim, since each
				# tuple contains all dim's. To save memory, just upload them on the 1st dim.
				if dim == 0: dataVec, placeVec = [], [] 
				prediction_tuples = []
				for i, data in enumerate(data_for_LM):
					segment, identity = inputs[i], indexes[i]
					data_points = segment[data][:, dim]
					process_string = segment[data-1]
					prediction_tuples += zip_(process_string, data_points)
					if dim == 0:
						dataVec += zip_(process_string, segment[data])
						placeVec += zip_(identity[data-1], identity[data])
				if dim == 0: 
					dataMat.append(dataVec)
					placeMat.append(placeVec)
				# Now I have the inputs and outputs
				tuplesVec.append(np.array(prediction_tuples))
			tuplesMat.append(tuplesVec)
		return tuplesMat, dataMat, placeMat

	# Useful function to calculate complete likelihoods across iters
	# (Hidden Function)
	def get_summed_likelihood(proba, model):
		summed_log_likelihood = 0
		for k in proba.keys():
			summed_log_likelihood += proba[k][model[k]]
		return summed_log_likelihood

	# Random Forest Forward Pass
	# --------------------------
	# An alternative to the forward-backward dirichlet. This uses Random
	# Forests to calculate probabilities. The procedure for running these
	# then becomes z --> randomize --> loop
	# forests = None for all of training. When it is testing, we will have 
	# to use the forests = forestMat (passed from training).
	def forward_randomize(tuples, num_trees=10):
		# Returns the summed log likelihood across dimensions.
		forestMat = [] # Stores forests.
		for lm in xrange(params.num_models):
			forestDim = []
			for dim in xrange(params.num_features):
				# This contains (x_in, x_out) tuples for 
				# a single dim given an LM.
				prediction_tuples = tuples[lm][dim]
				if len(prediction_tuples) > 0:
					# Split the string back into an array of chars
					forest_inputs     = np.array([list(i) for i in prediction_tuples[:, 0]])
					forest_outputs    = np.array(list(prediction_tuples[:, 1]))
					# Initialize the forest (with Warm start depending on last fit
					forest = ExtraTreesClassifier(n_estimators=num_trees, random_state=42)
					forest.fit(forest_inputs, forest_outputs)	
					# Append it to struct sorted by dim. (whether it is new or not).
					forestDim.append(forest)
				else: forestDim.append(None) # If we don't use this LM anymore, tell us.
			forestMat.append(forestDim)
		# forestMat is basically a forest of forests ;). 
		# You could say it's a rainforest.
		return forestMat

	# Random Forest Backward Pass
	# ---------------------------
	# Decay is a tuned exponential decay fxn with randomness for stochasticity
	def backward_randomize(data, places, models, forestMat, TMat, priorMat=None, decay=None, temper=[1,1]):
		# Initialize a function to calculate/update beta
		def get_log_prior():
			betas = np.array([1/float(params.num_models) for m in range(params.num_models)])
			# Add counts to the beta distribution from new layer
			for i in models:
				for j in i:
					betas[j] += 1
			# Now we can parametrize them
			dirichlets = npr.dirichlet(betas)
			parameters = np.log(dirichlets)
			return parameters

		# Since we don't want to risk the probabilities to be 0, we should
		# add something to it to help. It seems like naive bayes also reverse bags.
		# We want to this for an entire sequence array at a time.
		def reverse_bag(classOrd, probSeq):
			fullSeq = np.array([[1e-25 for i in V] for j in probSeq])
			fullSeq[:, classOrd] += probSeq
			fullSeq /= np.sum(fullSeq, axis=1).reshape((fullSeq.shape[0], 1))
			return fullSeq

		# Let's calculate the priors. This is the same as in the dirichlet model.
		prior = get_log_prior() if priorMat is None  else priorMat

		# Setup, we want to go through all the keys together
		total_data = flatten(data)
		total_places = flatten(places)

		# Prep some of the data to be easily fed into a forest Machine
		prepData = total_data
		inputs, outputs = prepData[:, 0] , prepData[:, 1] 
		currPlaces, nextPlaces = total_places[:, 0], total_places[:, 1]

		# Calculations for p(y_t | x_t)
		probDict = {}
		# Build an empty array containing the lm's 
		for i in nextPlaces:
			probDict[i] = np.zeros(params.num_models)
			
		for lm in xrange(params.num_models):
			for d in xrange(params.num_features):
				forest = forestMat[lm][d]
				if not forest is None:
					logproba = np.log(reverse_bag(forest.classes_, forest.predict_proba(inputs)))
					for p in xrange(len(logproba)):
						probDict[nextPlaces[p]][lm] += logproba[p][outputs[p][d]]
				else: # If we don't have any of this lm, then hardcode -inf.
					for p in xrange(len(nextPlaces)):
						probDict[nextPlaces[p]][lm] = np.log(0)
		# If we have a decay object, use it (the iteration is updated outside)
		if decay is not None: decay.apply(probDict) # in-place changing
		# Save a copy of the probDict since we need to return it.
		saveDict = copy.copy(probDict)
		for key in probDict.keys():
			saveDict[key] = probDict[key] + prior
		# Do a little prepping for the HMM run.
		# With the language model spread, We want to flatten it, 
		# then pick out only the ones in the placeArr, and ALSO 
		# Sort it so that it matches the dictionary later all.
		sortKey = total_places[:, 1]
		flatten_models = flatten(models)
		total_models = {}
		for i in sortKey:
			total_models[i] = flatten_models[i]

		# Calculate the summed likelihood before the HMM
		# pre_summed_log_likelihood = get_summed_likelihood(probDict, total_models)
		# Initialize a hidden Markov model to calculate smoothed posterior.
		hmm_forward, hmm_backward = hmm.build_hmm(params.num_models, TMat)
		forward_struct = hmm_forward(probDict, prior, temper)
		resampled, backward_struct = hmm_backward(forward_struct, total_models)
		# Calculate a total summed likelihood (finding the maximum theta likelihood throughout 
		# all lm's for each data vec). Intransigent to LM number.
		post_summed_log_likelihood = get_summed_likelihood(probDict, resampled)
		return resampled, prior, saveDict, post_summed_log_likelihood

	# Wrapper Forward Pass
	# ---------------------
	def forward(*args):
		return forward_randomize(*args)

	# Wrapper Backward Pass
	# ---------------------
	def backward(*args):
		return backward_randomize(*args)

	print("Successfully defined a Mixture of Language Models classifier with %d languages, %d features, %d words, and resampled using %s.\n" % (N, F, len(V), method))
	return preprocess, setup, forward, backward

# All backwards functions merely return a dictionary of probabilities,
# This function will take transform that dictionary into the model_format
# using train_indexes!
def update(indexes, models, resample):
	new_models = []
	for i, idx in enumerate(indexes):
		reform = np.array([resample[idx[j]] if (idx[j] in resample) else models[i][j] for j in range(len(idx))])
		new_models.append(reform)
	return new_models

# Also include an in-house resampling method here. This is an 
# alternative to HMM (just pure resample).
# Mainly used for debugging.
def direct_resample(probDict, states):
	resample = {} # Save the samples
	keys = sorted(probDict.keys())
	curP = np.array(keys)
	for curT in curP: # Loop through the probabilities
		sampled = hmm.smart_sampling(np.array([probDict[curT]]))
		resample[curT] = sampled[0]
	return resample

# Given the resulting probabilities, we want to convert those 
# into organized feature vectors (+ extract corresponding labels)
# This will be used in both training and testing.
def vectorize(probDict, inputs, targets, indexes, augment=True):
	keys = probDict.keys()
	# First thing is flatten them so that it makes it easy
	# for us to find things.
	flat_inputs = flatten(inputs)
	flat_targets = flatten(targets)
	flat_indexes = flatten(indexes)

	new_vectors, new_targets, new_indexes = [], [], []
	for key in keys:
		probVec   = list(probDict[key])
		hashedKey = flat_indexes == key
		if augment: # Augmentation of vector with real DATA.
			probVec += list(flat_inputs[hashedKey][0])
		# zeroth index since things should be unique
		targetVec = flat_targets[hashedKey][0]
		indexVec  = flat_indexes[hashedKey][0]
		# Append to the new objects
		new_vectors.append(probVec)
		new_targets.append(targetVec)
		new_indexes.append(indexVec)
	return np.array(new_vectors), np.array(new_targets), np.array(new_indexes)

# ------------------------------------------------------------------------------------
# Debugging / Visualization Tools for Language model analysis

# Probably useful between iterations. 
def show_language_spread(models):
	# Manipulate into a better moldable form
	tuplesToPlot = []
	for i, row in enumerate(models):
		for j, col in enumerate(row):
			tuplesToPlot.append([i, j, col])
	tuplesToPlot = np.array(tuplesToPlot)
	# Calculate colors
	norm = max(tuplesToPlot[:, 2])
	setup = np.array([float(i)/norm for i in tuplesToPlot[:, 2]])
	color = plt.cm.coolwarm(setup)
	# Do the actual plotting
	# sns.set_style("whitegrid")
	# define colors:  
	plt.scatter(tuplesToPlot[:, 0], tuplesToPlot[:, 1], marker='o', edgecolors=color,  s=150, linewidths=1, alpha=0.3)
	plt.ylim(0)
	plt.title('Language Model Spread')
	plt.show()

def diff_models(model1, model2):
	return sum([sum(~(i == j)) for i, j in zip(model1, model2)]) 

def get_language_target_spread(MLM):
	# Create the struct
	percentages = {}
	for i in range(N): percentages[i] = np.zeros(N)
	# Fill out the struct
	for i in range(len(MLM['targets'])):
		for j in range(len(MLM['targets'][i])):
			percentages[MLM['targets'][i][j]][MLM['dispersion'][i][j]] += 1
	# Calculate sums
	sums = [np.sum(percentages[i]) for i in range(N)]
	for i in range(N): percentages[i] /= sums[i]
	# Return the struct
	return percentages

