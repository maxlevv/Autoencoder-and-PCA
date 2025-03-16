import numpy as np
from tqdm import tqdm
from loss_activation import sigmoid, sigmoid_derivative, error


# file containing all functions that are needed to train a nonlinear autoencoder
# import the train_autoencoder_nonlinear function to train a nonlinear autoencoder


def forward_pass_nl(X, W1, W2):
    z1 = W1 @ X.T
    #o1 = sigmoid(z1)
    o1 = z1
    z2 = W2 @ o1
    o2 = sigmoid(z2)
    return W1, W2, z1, o1, z2, o2


def backpropagation(X, W1, W2, z1, o1, z2, o2):
    error2 = 2 * (o2 - X.T) * sigmoid_derivative(z2)
    gradient_batch = error2[:, np.newaxis, :] * o1[np.newaxis, :, :]
    dW2 = np.mean(gradient_batch, axis=2)

    #error1 = (W2.T @ error2) * sigmoid_derivative(z1)
    error1 = (W2.T @ error2)
    gradient_batch_tensor = error1[:, np.newaxis, :] * X.T[np.newaxis, :, :]
    dW1 = np.mean(gradient_batch_tensor, axis=2)
    return dW1, dW2


def update_weights(W1, W2, dW1, dW2, lr):
    W1 = W1 - lr * dW1
    W2 = W2 - lr * dW2
    return W1, W2


def train_autoencoder_nonlinear(X, hidden_layer_size:int, epochs:int, lr:float):
    W1 = (np.random.rand(hidden_layer_size, X.shape[1]) - 0.5) * 0.1
    W2 = (np.random.rand(X.shape[1], hidden_layer_size) - 0.5) * 0.1
    error_history = []
    for epoch in tqdm(range(epochs)):
        W1, W2, z1, o1, z2, o2 = forward_pass_nl(X, W1, W2)
        current_error = np.mean(error(X, o2), axis=0)
        print(current_error)
        error_history.append(current_error)

        dW1, dW2 = backpropagation(X, W1, W2, z1, o1, z2, o2)
        W1, W2 = update_weights(W1, W2, dW1, dW2, lr)

    return W1, W2, error_history
