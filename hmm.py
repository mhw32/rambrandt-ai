# Forward Filtering, Backward Sampling

from __future__ import absolute_import
from __future__ import print_function

import sys, copy
import numpy.random as npr
import numpy as np

class HMMParams(object):
	def __init__(self):
		self.num_states = 0
		self.T = np.zeros((0, 0))

	def set_num_states(self, num):
		self.num_states = num

	def set_T_matrix(self, matrix):
		S = self.num_states
		# T must be limited by N in size.
		T = np.zeros((S, S))
		for i in range(S):
			for j in range(S):
				T[i][j] = matrix[i][j]
		self.T = T

def build_hmm(S, T):
	params = HMMParams()
	params.set_num_states(S)
	params.set_T_matrix(T)

	# p(x_t | y_1, ..., y_t) ~ p(y_t, x_t) * sum(T(x_t, x_t-1) p(x_t-1 | y1, ..., y_t-1))
	def hmm_forward(likelihoods, prior, tempering=[1, 1]):
		# Extract tempering constants
		alpha, beta = tempering[0], tempering[1]
		# likelihood --> p(y_t, x_t) [ This has to be calculated elsewhere. ]
		# prior --> p(x_0) [ This must also be provided. ]
		# Both of the provided matrices are assumed to be in log space.
		forward = {}
		keys = sorted(likelihoods.keys())
		# Run the first iteration separately
		p1 = likelihoods[keys[0]]
		# Calculate log likelihood. This may be tempered.
		p2 = np.log(np.dot(params.T**beta, np.exp(prior)**alpha))
		forward[keys[0]] = lognormalize(p1 + p2)

		# Run the next iterations together
		lastp = forward[keys[0]] # Keep track of the last 
		curP = keys[1:] # Remove first iter
		for curT in curP: # Ignore the 1st iteration
			p1 = likelihoods[curT]
			p2 = np.log(np.dot(params.T, np.exp(lastp)))
			# Save the dictionary + save as counter 
			forward[curT] = lastp = lognormalize(p1 + p2)
		return forward

	# p(x_t | y1, ..., y_t, x_t+1) ~ p(x_t+1 | x_t) p(x_t | y_1, ..., y_t)
	# states - spread of states over data (in our case = language models over patients)
	def hmm_backward(forward, states):
		# forward.keys() should already be sorted.
		keys = sorted(forward.keys())[::-1] # reverse
		curP = np.array(keys[1:])
		nexP = curP + 1 

		lastp = keys[0]
		resample = {} # Save the samples
		backward = {}
		# Do the first iteration first
		sampled = smart_sampling(np.array([forward[lastp]]))
		resample[lastp] = sampled[0]
		backward[lastp] = forward[lastp]
		# Propogate the sampling. Consider vectorizing this operation.
		for curT, nexT in zip(curP, nexP):
			p1 = np.log(params.T[states[curT]]) # p(x_t+1 | x_t)
			p2 = forward[curT] # already in logspace
			result = lognormalize(p1 + p2)
			sampled = smart_sampling(np.array([result]))
			resample[curT] = sampled[0]
			backward[curT] = result
		return resample, backward

	return hmm_forward, hmm_backward
		
# Regular uniform sampling.
def sample(v, r):
	rolling = 0
	for i in xrange(len(v)):
		rolling += v[i]
		if r <= rolling:
			return i

# Sampling in log space with really small 
# numbers. A is an array of arrays. 
def smart_sampling(A):
	length = A.shape[0]
	# Important to normalize here.
	B = np.exp(A - np.max(A, axis=1).reshape((length, 1)))
	B /= np.sum(B, axis=1).reshape((length, 1))
	flips = npr.rand(length)
	return np.array([sample(b, r) for b,r in zip(B, flips)])
# ------------------------------------------------------------
# Code to generate a transition matrix based on gaussian 2D filter.

# We need to normalize these to sum to 1 in logspace. 
def lognormalize(x):
	a = np.logaddexp.reduce(x)
	return x - a

def normalize(A):
    return A / float(np.sum(A))

# 2 dimensional pdf of gaussian distribution
def gaussian_2d(muX, muY, sigma, x, y):
    return 1 / float(2*np.pi*sigma**2) * np.exp(-((x-muX)**2 + (y-muY)**2) / 2*sigma**2)

# Given some means and some coordinates, generate a filter
# Note that this returns a MATRIX.
def gaussian_2d_filter(muX, muY, minX, minY, maxX, maxY, sigma=1):
    F = np.zeros((maxX-minX, maxY-minY))
    for i in range(minX, maxX):
        for j in range(minY, maxY):
            F[i][j] = gaussian_2d(muX, muY, sigma, i, j)
    return F

def generate_2d_T_mat(n):
    T = np.zeros((n, n))
    # For each entry in the diagonal, generate a 2D gaussian
    for i in range(n):
        layer = gaussian_2d_filter(i, i, 0, 0, n, n)
        T = T + layer
    for i in range(n):
        T[i] = normalize(T[i])
    return T.T # transpose to sum to 1 column-wise

# 1 dimension pdf of gaussian distribution
def gaussian_1d(muX, sigma, x):
	return 1 / float(np.sqrt(2*np.pi) * sigma) * np.exp(-(x-muX)**2 / (2*sigma**2))

# Complement of 2d version. 
# Note that this returns a VECTOR.
def gaussian_1d_filter(muX, minX, maxX, sigma=1):
	F = np.zeros(maxX-minX)
	for i in range(minX, maxX):
		F[i] = gaussian_1d(muX, sigma, i)
	return F 

def generate_1d_T_mat(n):
	T = np.zeros((n, n))
	for i in range(n):
		T[i] = gaussian_1d_filter(i, 0, n)
		T[i] = normalize(T[i])
	return T.T # transpose to sum to 1 column-wise

# Create a wrapper function to the let user pick
# if they want 2-D filtering or 1-D filtering. 
def generate_T_mat(n, dim, *args):
	# Hard enforce edge values.
	if dim > 2: dim = 2 
	if dim == 0:
		return generate_0d_T_mat(n, *args)
	elif dim == 1:
		return generate_1d_T_mat(n, *args)
	else: # dim == 2
		return generate_2d_T_mat(n, *args)

# This is a hardcoded T matrix, in the sense 
# that we pick the diagonal values and set 
# all others to be such that the full "row"
# sums to 1. Useful for straight cutoffs.
def generate_0d_T_mat(n, bias=0.98): 
	T = np.zeros((n, n))
	fill = (1 - bias) / (n - 1)
	for i in range(n):
		T[i] = np.array([fill for j in range(n)])
		T[i][i] = bias
	return T.T # transpose


