import numpy as np
import os

# (PLEASE DO NOT CHANGE) Set random seed:
np.random.seed(1746)

PREFIX = "digit_"

TEST_STEM = "test_"
TRAIN_STEM = "train_"

def load_data(data_dir, stem):
    """
    Loads data from either the training set or the test set and returns the pixel values and
    class labels
    """
    data = []
    labels = []
    for i in range(0, 10):
        path = os.path.join(data_dir, PREFIX + stem + str(i) + ".txt")
        digits = np.loadtxt(path, delimiter=',')
        digit_count = digits.shape[0]
        data.append(digits)
        labels.append(np.ones(digit_count) * i)
    data, labels = np.array(data), np.array(labels)
    data = np.reshape(data, (-1, 64))
    labels = np.reshape(labels, (-1))
    return data, labels

def load_all_data(shuffle=True):
    '''
    Loads all data from the given data directory.

    Returns four numpy arrays:
        - train_data
        - train_labels
        - test_data
        - test_labels
    '''

    train_data, train_labels = load_data("", TRAIN_STEM)
    test_data, test_labels = load_data("", TEST_STEM)

    if shuffle:
        train_indices = np.random.permutation(train_data.shape[0])
        test_indices = np.random.permutation(test_data.shape[0])
        train_data, train_labels = train_data[train_indices], train_labels[train_indices]
        test_data, test_labels = test_data[test_indices], test_labels[test_indices]

    return train_data, train_labels, test_data, test_labels


def get_digits_by_label(digits, labels, query_label):
    '''
    Return all digits in the provided array which match the query label

    Input:
        - digits: numpy array containing pixel values for digits
        - labels: the corresponding digit labels (0-9)
        - query_label: the digit label for all returned digits

    Returns:
        - Numpy array containing all digits matching the query label
    '''
    assert digits.shape[0] == labels.shape[0]

    matching_indices = labels == query_label
    return digits[matching_indices]
