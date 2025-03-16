import numpy as np


def sigmoid(z: np.array):
    res = np.zeros_like(z)
    res[z > 0] = 1 / (1 + np.e**(-z[z > 0]))
    res[z <= 0] = np.exp(z[z <= 0]) / (1 + np.exp(z[z <= 0]))
    return res


def sigmoid_derivative(z: np.array):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z: np.array):
    # z (n x d)
    m = z.max(axis=0)
    trans_z = z - m[np.newaxis, :]
    e = np.exp(trans_z)  # (n x d)
    return e / e.sum(axis=0).reshape(1, -1)


def categorical_cross_entropy(h: np.array, y_one_hot: np.array):
    h = np.clip(h, a_min=0.000000001, a_max=None)

    return - (y_one_hot * np.log(h)).sum(axis=1).mean(axis=0)


def d_categorical_cross_entropy_with_softmax(z: np.array, y_one_hot: np.array):
    # z (n x b); y_one_hot (b x n)
    # res (b x n)
    return softmax(z).T - y_one_hot


def error(X, output):
    return np.mean(np.power(output - X.T, 2), axis=0)


def error_derivative(X, output):
    return np.sum((output - X.T), axis=0)


def one_hot_encoding(y: np.array):
    y_g = np.zeros((len(y), 10))
    for i in range(len(y)):
        j = y[i]
        y_g[i, j] = 1
    return y_g