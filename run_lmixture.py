import numpy as np
import numpy.random as npr
import lmixture as lm
import hmm
import cross_validation as cv2
# Wrapper scripts to implement mixture of language models. 

# If stochastic is true, then set it up here
# decayFuncs = setup_stochasticity(saveProbs)
# saveProbs = add_stochasticity(saveProbs, decayFuncs, 1)

# Wrapper function to set up individual test and training fxns.
# The user is given the option to choose a input their own classifier
def run_lmixture(N, F, V, R):
	# Initialize the lmixture object (different fxns depending on method used.)
	preprocess_fun, setup_fun, forward_fun, backward_fun = lm.build_lmixture(N, F, V, R)
	T = hmm.generate_T_mat(N, 2) # Transition matrix

	def train_lmixture(X_train, y_train, I_train, maxIter=200, epsilon=0.001, stochastic=False, seed=1, tempering=[1,1]):
		# Create a decay object (might or might not be used) but this is a 1 time cost 
		decayObj = None # Default 
		if stochastic:
			decayObj = lm.MixtureDecay(seed=seed, maxiter=maxIter)
			decayObj.update_iter(1)
		# Save the likelihooods
		likelihoods_over_time = [] 
		train_inputs, train_targets, train_models, train_indexes, train_raw = \
			preprocess_fun(X_train, y_train, I_train, 'random')
		# Run the first iteration separately. Regardless of R, all of these are the same.
		tupleMat, dataMat, placeMat = setup_fun(train_inputs, train_indexes, train_models)
		classMat = forward_fun(tupleMat)
		resample, priorProbs, saveProbs, logsum = backward_fun(dataMat, placeMat, train_models, classMat, T, None, decayObj, tempering)
		# Go back to the model updates
		train_models = lm.update(train_indexes, train_models, resample)
		likelihoods_over_time.append(logsum)
		print("Iteration: %d." % 1)
		
		# Run the rest of the iterations.
		currIter = 2
		while (currIter < maxIter):
			# If there is a decayObj, then we need to update the iteration
			if stochastic: decayObj.update_iter(currIter)
			# Continue applying the language model for subsequence iterations.
			tupleMat, dataMat, placeMat = setup_fun(train_inputs, train_indexes, train_models)
			classMat = forward_fun(tupleMat)
			resample, priorProbs, saveProbs, logsum = backward_fun(dataMat, placeMat, train_models, classMat, T, None, decayObj, tempering)
			# Go back to the model updates
			train_models = lm.update(train_indexes, train_models, resample)
			likelihoods_over_time.append(logsum)
			print("Iteration: %d." % currIter)
			currIter += 1

		# Create vectors for classifier training
		clf_input, clf_output, _ = lm.vectorize(saveProbs, train_raw, train_targets, train_indexes, True)
		MLM = {'train-inputs':clf_input,
			   'train-targets':clf_output,
			   'train-likelihoods':likelihoods_over_time,
			   'classifiers':classMat,
			   'prior':priorProbs,
			   'tempering':tempering}
		# return the object of interest.
		return MLM

	# We are assuming that at this point, we also have the functions 
	# (preprocess, setup, forward, backward)
	def test_lmixture(X_test, y_test, I_test, MLM):
		classMat          = MLM['classifiers']
		prior             = MLM['prior']
		clf_train_inputs  = MLM['train-inputs']
		clf_train_targets = MLM['train-targets']
		tempering         = MLM['tempering']

		# Preprocess the test framework.
		test_inputs, test_targets, test_models, test_indexes, test_raw = \
			preprocess_fun(X_test, y_test, I_test)
		tupleMat, dataMat, placeMat = setup_fun(test_inputs, test_indexes, test_models)
		# Run the backwards fun
		resample, priorProbs, saveProbs, logsum = backward_fun(dataMat, placeMat, test_models, classMat, T, prior, None, tempering)
		test_models = lm.update(test_indexes, test_models, resample)
		clf_test_input, clf_test_output, _ = lm.vectorize(saveProbs, test_raw, test_targets, test_indexes, True)
		# Add more attributes to the wrapper object
		MLM['test-inputs'] = clf_test_input
		MLM['test-targets'] = clf_test_output
		MLM['test-likelihood'] = logsum
		MLM['dispersion'] = test_models
		return MLM

	return train_lmixture, test_lmixture

def run_classifiers(MLM, disp=False):
	X_train = MLM['train-inputs']
	y_train = MLM['train-targets']
	X_test = MLM['test-inputs']
	y_test = MLM['test-targets']
	# Run the standard classiiers.
	cv2.run_testing_suite(X_train, y_train, X_test, y_test, disp=disp)
	return 0








				








