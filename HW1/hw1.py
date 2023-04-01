# These are imports and you do not need to modify these.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import sklearn
import random
import math

# ===================================================
# You only need to complete or modify the code below.

def process_data(data, labels):
	"""
	Preprocess a dataset of strings into vector representations.

    Parameters
    ----------
    	data: numpy array
    		An array of N strings.
    	labels: numpy array
    		An array of N integer labels.

    Returns
    -------
    train_X: numpy array
		Array with shape (N, D) of N inputs.
    train_Y:
    	Array with shape (N,) of N labels.
    val_X:
		Array with shape (M, D) of M inputs.
    val_Y:
    	Array with shape (M,) of M labels.
    test_X:
		Array with shape (M, D) of M inputs.
    test_Y:
    	Array with shape (M,) of M labels.
	"""
	
	# Split the dataset of string into train, validation, and test 
	# Use a 70/15/15 split
	# train_test_split shuffles the data before splitting it 
	# Stratify keeps the proportion of labels the same in each split

	# -- WRITE THE SPLITTING CODE HERE -- 
	train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(data, labels, test_size=0.30, train_size=0.70, random_state=None, shuffle=True, stratify=labels)
	val_X, test_X, val_Y, test_Y = sklearn.model_selection.train_test_split(test_X, test_Y, test_size=0.5, train_size=0.5, random_state=None, shuffle=True, stratify=test_Y)


	# Preprocess each dataset of strings into a dataset of feature vectors
	# using the CountVectorizer function. 
	# Note, fit the Vectorizer using the training set only, and then
	# transform the validation and test sets.

	# -- WRITE THE PROCESSING CODE HERE -- 
	c = CountVectorizer()
	c.fit(train_X)
	train_X = c.transform(train_X)
	val_X = c.transform(val_X)
	test_X = c.transform(test_X)
	# # Return the training, validation, and test set inputs and labels

	# # -- RETURN THE ARRAYS HERE -- 

	return train_X, train_Y, val_X, val_Y, test_X, test_Y

def select_knn_model(train_X, val_X, train_Y, val_Y):
	"""
	Test k in {1, ..., 20} and return the a k-NN model
	fitted to the training set with the best validation loss.

    Parameters
    ----------
    	train_X: numpy array
    		Array with shape (N, D) of N inputs.
    	train_X: numpy array
    		Array with shape (M, D) of M inputs.
    	train_Y: numpy array
    		Array with shape (N,) of N labels.
    	val_Y: numpy array
    		Array with shape (M,) of M labels.

    Returns
    -------
    best_model : KNeighborsClassifier
    	The best k-NN classifier fit on the training data 
    	and selected according to validation loss.
  	best_k : int
    	The best k value according to validation loss.
	"""
	best_k = None
	best_model = None
	best_score = None

	for k in range(1,21):
		n = KNeighborsClassifier(n_neighbors=k)
		n.fit(train_X,train_Y)
		ypredict=n.predict(val_X)
		X = ypredict.reshape(-1, 1)
		Y = val_Y.reshape(-1,1)
		z = n.score(val_X,val_Y)
		if best_score == None or z < best_score:
			best_score = z
			best_k = k
			best_model = n

	return best_model,best_k



# You DO NOT need to complete or modify the code below this line.
# ===============================================================


# Set random seed
np.random.seed(3142021)
random.seed(3142021)

def load_data():
	# Load the data
	with open('./clean_fake.txt', 'r') as f:
		fake = [l.strip() for l in f.readlines()]
	with open('./clean_real.txt', 'r') as f:
		real = [l.strip() for l in f.readlines()]

	# Each element is a string, corresponding to a headline
	data = np.array(real + fake)
	labels = np.array([0]*len(real) + [1]*len(fake))
	return data, labels


def main():
	data, labels = load_data()
	train_X, train_Y, val_X, val_Y, test_X, test_Y = process_data(data, labels)


	best_model, best_k = select_knn_model(train_X, val_X, train_Y, val_Y)
	test_accuracy = best_model.score(test_X, test_Y)
	print("Selected K: {}".format(best_k))
	print("Test Acc: {}".format(test_accuracy))


if __name__ == '__main__':
	main()
