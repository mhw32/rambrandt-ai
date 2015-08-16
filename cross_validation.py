# A Stanrdard Cross Validation compilation. 
# Written by Scikit-Learn

# 1. Split your data into training and testing (80/20 is indeed a good starting point)
# 2. Split the training data into training and validation (again, 80/20 is a fair split).
# 3. Subsample random selections of your training data, train the classifier with this, 
#    and record the performance on the validation set
# 4. Try a series of runs with different amounts of training data: 
#    randomly sample 20% of it, say, 10 times and observe performance on the validation data, 
#    then do the same with 40%, 60%, 80%. You should see both greater performance with more data, 
#    but also lower variance across the different random samples
# 5. To get a handle on variance due to the size of test data, perform the same procedure 
#    in reverse. Train on all of your training data, then randomly sample a percentage of 
#    your validation data a number of times, and observe performance. 
#    You should now find that the mean performance on small samples of your validation data
#    is roughly the same as the performance on all the validation data, but the variance is 
#    much higher with smaller numbers of test samples.

from sklearn import cross_validation as cv
import pandas as pd, numpy as np
import pickle, os, sys, datetime
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from random import randint, shuffle
import z_scored_words as zs
import matplotlib.pyplot as plt

RAND_SEED = 42 # So that we can repeat :)
def eval_pos_neg(ytest, yhat):
	false_positive = 0
	false_negative = 0
	true_positive = 0
	true_negative = 0

	for (i,j) in zip(yhat, ytest):
		if j == 0:
			if i == 0: true_negative += 1
			else: false_positive += 1
		else: # j == 1
			if i == 1: true_positive += 1
			else: false_negative += 1
				
	print "False Positives", false_positive
	print "False Negatives", false_negative
	print "True Positives", true_positive
	print "True Negatives", true_negative

def run_testing_suite(X_train, y_train, X_test, y_test, disp=False):
	# Move test train split to be separate from the model choice
	run_standard_SVM(X_train, X_test, y_train, y_test, disp)
	run_standard_LogReg(X_train, X_test, y_train, y_test, disp)
	run_standard_NBayes(X_train, X_test, y_train, y_test, disp)	
	run_standard_KNN(X_train, X_test, y_train, y_test, disp)

def run_standard_SVM(X_train, X_test, y_train, y_test, disp=True, graph=False):
	clf = svm.SVC()
	clf.fit(X_train, y_train)  
	y_pred = clf.predict(X_test)
	if disp: eval_pos_neg(y_test, y_pred)
	
	y_score = clf.decision_function(X_test)
	fpr, tpr, _ = roc_curve(y_test, y_score)
	if graph == True:
		plt.plot(fpr, tpr)
		plt.title("SVM Decision Function ROC")
		plt.grid(True)
		plt.show()
	print "SVM (AUC): %f" % auc(fpr, tpr)

def run_standard_LogReg(X_train, X_test, y_train, y_test, disp=True, graph=False):
	clf = LogisticRegression()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	if disp: eval_pos_neg(y_test, y_pred)

	y_score = clf.decision_function(X_test)
	fpr, tpr, _ = roc_curve(y_test, y_score)
	if graph == True:
		plt.plot(fpr, tpr)
		plt.title("LogReg Decision Function ROC")
		plt.grid(True)
		plt.show()
	print "LogReg (AUC): %f" % auc(fpr, tpr)

def run_standard_NBayes(X_train, X_test, y_train, y_test, disp=True, graph=False):
	gnb = GaussianNB()
	gnb.fit(X_train, y_train)
	y_pred = gnb.predict(X_test)
	if disp: eval_pos_neg(y_test, y_pred)
	y_proba = gnb.predict_proba(X_test)
	y_score = y_proba[:,1]
	fpr, tpr, _ = roc_curve(y_test, y_score)
	if graph == True:
		plt.plot(fpr, tpr)
		plt.title("NBayes Probabilities ROC")
		plt.grid(True)
		plt.show()
	print "NBayes (AUC): %f" % auc(fpr, tpr)

def run_standard_KNN(X_train, X_test, y_train, y_test, disp=True, graph=False):
	neigh = KNeighborsClassifier(n_neighbors=2)
	neigh.fit(X_train, y_train)
	y_pred = neigh.predict(X_test) 
	if disp: eval_pos_neg(y_test, y_pred)
	
	y_proba = neigh.predict_proba(X_test)
	y_score = y_proba[:,1]
	fpr, tpr, _ = roc_curve(y_test, y_score)
	if graph == True:
		plt.plot(fpr, tpr)
		plt.title("NNeighbors Probabilities ROC")
		plt.grid(True)
		plt.show()
	print "KNN (AUC): %f" % auc(fpr, tpr)

def subsampling_split(X_train, y_train, I_train):
	# Now given X_train, y_train, we want to make sure its
	# around 70%-30% ratio, so that we don't make it 
	# overbearing for the classifier
	goodfraction = 0.30
	badfraction = 1 - goodfraction

	goodcount = sum(y_train) # this tells us how many 1's there are
	badcount = len(y_train) - goodcount # this is how many 0's there are
	if badcount <= int(goodcount * badfraction / float(goodfraction)): 
		# In this case there are enough good counts in ratio to bad counts
		e_X_train = X_train
		e_y_train = y_train
		e_I_train = I_train
	else: # If this case, there are too many bad counts, so we have to sample
		num2resample = int(goodcount * badfraction / float(goodfraction)) + 1
		# Let's physically separate the parts for the training set
		goodbool = (y_train == 1)
		badbool = np.invert(goodbool)
		goodX = X_train[goodbool]
		goody = y_train[goodbool]
		goodI = I_train[goodbool]
		badX = X_train[badbool]
		bady = y_train[badbool]
		badI = I_train[badbool]
		# define a useful parameter
		resampleSize = len(badX)

		# Draw a sample for each value in array
		e_X_train = []
		e_y_train = []
		e_I_train = []
		for i in range(num2resample):
			RV = randint(0, resampleSize - 1)
			sampleX = badX[RV]
			sampley = bady[RV]
			sampleI = badI[RV]
			e_X_train.append(sampleX)
			e_y_train.append(sampley)
			e_I_train.append(sampleI)

		# combine these with the good ones
		e_X_train = e_X_train + list(goodX)
		e_y_train = e_y_train + list(goody)
		e_I_train = e_I_train + list(goodI)

		# Shuffle the two lists together to randomize structure
		combined = zip(e_X_train, e_y_train, e_I_train)
		shuffle(combined)
		e_X_train[:], e_y_train[:], e_I_train = zip(*combined)

		# finally, convert to numpy array
		e_X_train = np.array(e_X_train)
		e_y_train = np.array(e_y_train)
		e_I_train = np.array(e_I_train)

	return e_X_train, e_y_train, e_I_train


# This function helps us split training and testing data 
# in a more intelligent manner ensuring that complete patient data
# will be in one set.
def train_test_split_by_patient(FV, L, I, subsample=True):
	# Start the training testing smart split
	uniqueI = np.unique(np.array(I))
	uniqueL = []
	# Given each unique SUBJECT 
	for patient in uniqueI:
		relevantI = I[I == patient]
		relevantL = L[I == patient]
		# There must be at least one positive label
		tmp = 1 if sum(relevantL) > 0 else 0 
		uniqueL.append(tmp)
	uniqueL = np.array(uniqueL)
	
	# Now use scikit learn's splits on these
	# I don't think the 2nd two things matter here...
	I_train, I_test, _, _ = cv.train_test_split(uniqueI, uniqueL, test_size=0.25, random_state=42)
	
	# Now we can use these to split the real sets.
	X_train, X_test = [], []
	y_train, y_test = [], []
	full_I_train, full_I_test = [], []
	for i in I_train:
		X_train += list(FV[I == i])
		y_train += list(L[I == i])
		full_I_train += [i for thing in range(len(list(FV[I == i])))]
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	full_I_train = np.array(full_I_train)

	for j in I_test:
		X_test += list(FV[I == j])
		y_test += list(L[I == j])
		full_I_test += [j for thing in range(len(list(FV[I == j])))]
	X_test = np.array(X_test)
	y_test = np.array(y_test)
	full_I_test = np.array(full_I_test)
	
	# Do the subsampling
	if subsample:
		X_train, y_train, full_I_train = subsampling_split(X_train, y_train, full_I_train)
	
	return X_train, y_train, X_test, y_test, full_I_train, full_I_test

def training_testing_separate_interpolate(X_train, y_train, X_test, y_test, allFeatures, contFeatures):
	# Grab the number of features
	num_of_features = X_train.shape[1]
	# Find the indexes of the interesting features
	continuous_features = [np.where(np.array(allFeatures) == i)[0][0] for i in contFeatures]

	# Placeholders for means
	feature_means = np.zeros(num_of_features)
	feature_stds = np.zeros(num_of_features)
	# Fill out distribution mean and std from training data
	for f in continuous_features: # For the categorical ones, assume 0
		# Calculate population stats from non-np data
		non_nan_data =  X_train[:, f][~np.isnan(list(X_train[:, f]))]
		non_nan_data = non_nan_data.tolist() 	
		feature_means[f] = np.mean(non_nan_data)
		feature_stds[f] = np.std(non_nan_data)

	# Fill out the nan's (replacing the first ones with population means..
	X_train = sample_and_hold(X_train, feature_means)

	# For each continuous feature, just replace it with the z-score
	for f in continuous_features:
		X_train[:, f] = zs.z_score(X_train[:, f], feature_means[f], feature_stds[f])

	# Use the calculated info to apply it on the testing data
	# First, fill out the Nan's with population mean's from the training
	X_test = sample_and_hold(X_test, feature_means)
	# Then, z-score it with the training set mean and std
	for f in continuous_features:
		X_test[:, f] = zs.z_score(X_test[:, f], feature_means[f], feature_stds[f])

	return X_train, y_train, X_test, y_test


def training_testing_complete_interpolate(X_train, y_train, X_test, y_test, \
		staticFeatures, contStaticFeatures, variableFeatures, LoopNumber=4):
	# First let's make note of all the features (with looping involved)
	totalFeatures = list(np.array(staticFeatures).copy())
	for i in range(LoopNumber):
		totalFeatures += variableFeatures # Since we do 4 hour

	# We should find the indices of the static features in the total features
	boolContStaticFeatures = findListInList(totalFeatures, contStaticFeatures)
	# Find mean and std for static variables first
	static_feature_means = np.zeros(len(staticFeatures))
	static_feature_stds = np.zeros(len(staticFeatures))
	# For static features, process the means regularly
	for f in boolContStaticFeatures:
		# Calculate population stats from non-np data
		static_feature_means[f] = np.mean(X_train[:, f][~np.isnan(list(X_train[:, f]))])
		static_feature_stds[f] = np.std(X_train[:, f][~np.isnan(list(X_train[:, f]))])

	# Now we have to calculate the means and std for variables ones (trickier)
	variable_feature_means = np.zeros(len(variableFeatures))
	variable_feature_stds = np.zeros(len(variableFeatures))
	idx = 0
	# This is for remembering locations of the features in array
	totalBoolVariableFeatures = np.zeros(len(totalFeatures))
	for f in variableFeatures:
		# Generate the boolean array (find where the feature is)
		boolFeatures = np.zeros(len(totalFeatures))
		boolFeatures = np.logical_or(boolFeatures, np.array(totalFeatures) == f)
		# Grab the indices for those features
		boolVariableFeatures = np.array(range(len(totalFeatures)))[boolFeatures]
		totalBoolVariableFeatures = np.logical_or(totalBoolVariableFeatures, boolFeatures)
		# merge the lists (all instance of the feature)
		merged = np.array([list(X_train[:, i]) for i in boolVariableFeatures]).flatten()
		variable_feature_means[idx] = np.mean(merged[~np.isnan(merged)])
		variable_feature_stds[idx] = np.std(merged[~np.isnan(merged)])
		if np.isnan(variable_feature_means[idx]):			
			variable_feature_means[idx] = 0  
		if np.isnan(variable_feature_stds[idx]):			
			variable_feature_stds[idx] = 0  
			
		idx += 1 # Count ticker

	# Combine them together into a complete mean/std 
	total_feature_means = list(static_feature_means)
	total_feature_stds = list(static_feature_stds)
	for i in range(LoopNumber):
		total_feature_means += list(variable_feature_means)
		total_feature_stds += list(variable_feature_stds)
	total_feature_means = np.array(total_feature_means)

	# GREAT! now we are reading to proceed. 
	# Sample and hold for the training 
	X_train = sample_and_hold(X_train, total_feature_means)

	# Make a list of all indicies the z-scoring features
	features_zscore = list(boolContStaticFeatures) + list(np.array(range(len(totalFeatures)))[totalBoolVariableFeatures])

	# Z-score everything in the training set
	for f in features_zscore:
		X_train[:, f] = zs.z_score(X_train[:, f], total_feature_means[f], total_feature_stds[f])

	# Now we can do the same for the testing set (WITH TRAINING calculations)
	X_test = sample_and_hold(X_test, total_feature_means)
	# Then, z-score it with the training set mean and std
	for f in features_zscore:
		X_test[:, f] = zs.z_score(X_test[:, f], total_feature_means[f], total_feature_stds[f])

	return X_train, y_train, X_test, y_test

def findListInList(BigList, SmallList):
	return [i for i, x in enumerate(BigList) if any(thing in x for thing in SmallList)]


# [[...], [...], [...], [...], ...]
# To deal with NaN's
def sample_and_hold(nestArr, popMean):
	# if we z-score, the population mean should be 0's... 
	tracker = popMean.copy()
	for inner in nestArr:
		for cov in range(len(inner)):
			# If NaN, replace it with saved value
			if np.isnan(inner[cov]): 
				inner[cov] = tracker[cov]
			else: # If not NaN, save it
				tracker[cov] = inner[cov]
	return nestArr
