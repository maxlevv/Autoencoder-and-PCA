import numpy as np
from tqdm import tqdm
from loss_activation import error


# file containing all functions that are needed to train a linear autoencoder
# import the train_autoencoder_linear function to train a linear autoencoder


def forward_pass_l(X, W1, W2):
    input = X.T
    o1 = W1 @ input
    o2 = W2 @ o1
    return W1, W2, o1, o2


def get_gradients(X, W1, W2):
    average_factor = np.shape(X)[1]
    XXT = X.T @ X
    W2W1_I = (W2 @ W1) - np.identity(np.shape(W2)[0])
    dW1 = (W2.T @ W2W1_I @ XXT) / average_factor

    dW2 = W2W1_I @ XXT @ W1.T / average_factor
    return dW1, dW2


def backpropagation(X, W1, W2, o1, o2):
    error2 = 2 * (o2 - X.T)
    gradient_batch = error2[:, np.newaxis, :] * o1[np.newaxis, :, :]
    dW2 = np.mean(gradient_batch, axis=2)

    error1 = W2.T @ error2
    gradient_batch = error1[:, np.newaxis, :] * X.T[np.newaxis, :, :]
    dW1 = np.mean(gradient_batch, axis=2)
    return dW1, dW2


def update_weights(W1, W2, dW1, dW2, lr):
    W1 = W1 - lr * dW1
    W2 = W2 - lr * dW2
    return W1, W2


def train_autoencoder_linear(X, hidden_layer_size:int, epochs:int, lr:float):
    W1 = (np.random.rand(hidden_layer_size, X.shape[1]) - 0.5) * 0.1
    W2 = (np.random.rand(X.shape[1], hidden_layer_size) - 0.5) * 0.1
    error_history = []
    for epoch in tqdm(range(epochs)):
        W1, W2, o1, o2 = forward_pass_l(X, W1, W2)
        current_error = np.mean(error(X, o2), axis=0)
        print(current_error)
        error_history.append(current_error)

        dW1, dW2 = get_gradients(X, W1, W2)
        W1, W2 = update_weights(W1, W2, dW1, dW2, lr)

    return W1, W2, error_history
