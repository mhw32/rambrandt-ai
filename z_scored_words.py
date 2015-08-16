# Given data of covariates, this is intended to take the data stored 
# inside the Pandas structure and then classify it into groups (time-based)
# and then classify each group as a label

# Function 1: Given a sequence of data --> split into bins of data split on time 
# --- TO DO ---
import numpy as np

# Function 2: Given the bins, aggregate each bin with some pooling algorithm
def aggregate_bins(nestedBinArray, f): # arbitrary fxn "f"
    # do something for each nested bin (2D array)
    return [f(i) for i in nestedBinArray]

# Function 3: Given an array of bins, create a normal distribution around it
def create_gaussian_distr(binArray):
    meanArr, stdArr = np.mean(binArray), np.std(binArray)
    return meanArr, stdArr

def z_score(x, mu, sigma):
    return (x - mu) / float(sigma)

def p_value(x, mu, sigma):
    return 1 / (float(np.sqrt(2*np.pi))*sigma) * np.exp**(-(x - mu)**2/ float(2*sigma**2))

# Function 4 : Given the array of bins, calculate an array of z-scores
def discretize(array):
    # discretize the z_scores
    d_array = [int(round(i, 0)) for i in array]
    return np.array(d_array)
    
# Subfunction : Some labels to essentially
def string_label_zscores(zscore):
    # Two dimensions of descriptions
    # [lower, middle, upper]
    # [low, center, high]
    if zscore in [-1, 0, 1]:
        return "e"
    elif zscore == 2:
        return "f"
    elif zscore == -2:
        return "d"
    elif zscore == 3:
        return "g"
    elif zscore == -3:
        return "c"
    elif zscore == 4:
        return "h"
    elif zscore == -4:
        return "b"
    elif zscore >= 5:
        return "i"
    elif zscore <= -5:
        return "a"

def alphabetize(nestedArray):
    tmp = []
    for row in nestedArray:
        tmp2 = []
        for val in row:
            tmp2.append(string_label_zscores(val))
        tmp.append(tmp2)
    return np.array(tmp)


