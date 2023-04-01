"""STA314 Homework 3.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking STA314 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.
"""


from utils import *

import matplotlib.pyplot as plt
import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    #####################################################################

    #TODO: am i dealing with the dummy correctly?
    n = np.shape(data)[0]
    ones_vector = np.ones((n, 1))
    X = np.concatenate((data, ones_vector), axis=1)

    z = X.dot(weights) 
    y = 1.0 / (1.0 + np.exp(-z))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          D is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    #####################################################################

    n = np.shape(y)[0]
    ce = (-1*(targets.T.dot(np.log(y)))) / n 
    predictions = np.where(y >= 0.5,1.,0.)
    frac_correct = (predictions == targets).mean()


    #frac_correct = 1-((np.count_nonzero(targets - predictions))/n)
    #frac_correct = (np.sum(predictions == targets))/n
    #frac_correct = (np.sum(predictions == targets))/len(predictions == targets)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (D+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    lambd = hyperparameters["weight_regularization"]
    #####################################################################
    # TODO:                                                             #
    #####################################################################
    n = np.shape(data)[0]
    D = np.shape(data)[1]
    dummy_column = np.ones((n, 1))
    X = np.concatenate((data, dummy_column), axis=1)
    ce, frac_correct = evaluate(targets,y)
    f =  ce + (lambd/2)*(weights[0:D].T.dot(weights[0:D]))
    c = 1/n
    #predictions = np.where(y >= 0.5,1.,0.)
    dif = y - targets
    dw = np.dot(X.T, dif)
    dw = c*dw
    dummy_weight = np.zeros((1, 1))
    penality_deriv = np.concatenate((lambd*weights[0:D], dummy_weight), axis=0)
    df = dw + penality_deriv

    #new thing
    # a = np.dot(X, weights)
    # b = a-y
    # df = np.dot(X.T,b) + penality_deriv

    #df = dw + (lambd*weights[0:D]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


def run_logistic_regression():
    # Load all necessary datasets:
    x_train, y_train = load_train()
    # If you would like to use digits_train_small, please uncomment this line:
    #x_train, y_train = load_train_small()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    n, d = x_train.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations                                                     #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.001,
        "num_iterations": 1000
    }
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # compute test error, etc ...                                       #
    #####################################################################
    weights = np.zeros((d + 1, 1))
    alpha = hyperparameters["learning_rate"]
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, x_train, y_train, hyperparameters)
        weights = weights - alpha*df

    # Print the cross-entropy and accuracy for training set.
    y_pred = logistic_predict(weights, x_train)
    crossentropy,correct_frac = evaluate(y_train, y_pred)
    print("Training set cross-entropy: {}".format(crossentropy[0][0]))
    print("Training set accuracy: {}".format(correct_frac))
    # Print the cross-entropy and accuracy for validation set
    y_pred = logistic_predict(weights, x_valid)
    crossentropy,correct_frac = evaluate(y_valid, y_pred)
    print("Validation set cross-entropy: {}".format(crossentropy[0][0]))
    print("Validation set accuracy: {}".format(correct_frac))
    # Print the cross-entropy and accuracy for test set
    y_pred = logistic_predict(weights, x_test)
    crossentropy,correct_frac = evaluate(y_test, y_pred)
    print("Test set cross-entropy: {}".format(crossentropy[0][0]))
    print("Test set accuracy: {}".format(correct_frac))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_logistic_regression()
