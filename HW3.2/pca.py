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
import scipy.linalg as lin
import numpy as np


def pca(x, k):
    """ PCA algorithm. Given the data matrix x and k,
    return the eigenvectors, mean of x, and the projected data (code vectors).

    Hint: You may use NumPy or SciPy to compute the eigenvectors/eigenvalues.

    :param x: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :param k: int
        Number of dimension to reduce to.
    :return: Tuple of (Numpy array, Numpy array, Numpy array)
        WHERE
        v: A matrix of dimension D x k that stores top k eigenvectors
        mean: A vector of dimension D x 1 that represents the mean of x.
        proj_x: A matrix of dimension k x N where x is projected down to k dimension.
    """
    n, d = x.shape
    #####################################################################
    # TODO:                                                             #
    #####################################################################
    mean = np.mean(x, axis=0)
    a = x - mean.reshape(-1,1).T
    b = a.T
    cov = (1/n)*(b.dot(a))
    w,v = lin.eigh(cov,eigvals=(d-k,d-1))
    proj_x = v.T.dot(a.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return v, mean, proj_x


def show_eigenvectors(v):
    """ Display the eigenvectors as images.
    :param v: NumPy array
        The eigenvectors
    :return: None
    """
    plt.figure(1)
    plt.clf()
    for i in range(v.shape[1]):
        plt.subplot(1, v.shape[1], i + 1)
        plt.imshow(v[:, v.shape[1] - i - 1].reshape(16, 16).T, cmap=plt.cm.gray)
    plt.show()


def pca_classify():
    # Load all necessary datasets:
    x_train, y_train = load_train()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    # Make sure the PCA algorithm is correctly implemented.
    v, mean, proj_x = pca(x_train, 5)
    # The below code visualize the eigenvectors.
    show_eigenvectors(v)

    #####################################################################
    # TODO:                                                             #
    #####################################################################
    k_lst = [2, 5, 10, 20, 30]
    val_acc = np.zeros(len(k_lst))
    for j, k in enumerate(k_lst):
        v, mean, proj_x = pca(x_train, k)
        p = proj_x.T
        loss = 0
        for i in range(x_valid.shape[0]):
            # For each validation sample, perform 1-NN classifier on
            # the training code vector.
            project_i = v.T.dot((x_valid[i] - mean).T)
            smallest_distance = None
            closet_index = None
            for l in range(p.shape[0]):
                d = np.linalg.norm(p[l]-project_i)
                if smallest_distance == None or d < smallest_distance:
                    smallest_distance = d
                    closet_index = l
            label = y_train[closet_index]
            if label != y_valid[i]:
                loss = loss + 1
        print("For K = %s Accuracy = %s" % (k, 100*(1-(loss/x_valid.shape[0]))))
        val_acc[j]=100*(1-(loss/x_valid.shape[0]))
    f,p = plt.subplots()
    p.plot(k_lst, val_acc, 'b', label="Validation Set", marker="o")
    for m, n in enumerate(k_lst):
        s = "K = %s" % (n)
        p.annotate(s, (n, val_acc[m]))
    title = 'Num. Eigenvectors (K) v.s. Accuracy for Validation Set (%)'
    p.set(xlabel='Num. Eigenvectors (K)', ylabel='Accuracy', title=title)
    p.grid()
    leg = plt.legend(loc='lower right')
    plt.savefig('q3b.png', dpi=300, bbox_inches='tight')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    plt.plot(k_lst, val_acc)
    plt.show()


if __name__ == "__main__":
    pca_classify()
