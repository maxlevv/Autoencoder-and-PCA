import numpy as np
from mnist import MNIST
from mnist_helper import mnist_downloader
from pca import pca
from plots import plot_reconstructions_2, plot_embeddings_2d
from l_autoencoder import train_autoencoder_linear, forward_pass_l


# file executes pca with 2 dimension and trains a linear autoencoder with 2 hidden layer units
# plot both embeddings and reconstructions


# MNIST import
download_folder = r"mnist_helper/data"
mnist_downloader.download_and_unzip(download_folder)
mndata = MNIST(download_folder, return_type="numpy")
images_train, labels_train = mndata.load_training()
images_train = images_train[0:3000, :]
X = images_train
X = X/255
data_mean = np.mean(X, axis=0)


# PCA calculation
dimension = 2
pca_data, loading_matrix = pca(X.T, dimension=dimension, scree_plot=False)       # dimension x data_samples
reconstructed_pca_data = loading_matrix @ pca_data + data_mean[:, np.newaxis]


# Train linear AE
hidden_layer_size = 2
W1, W2, error_history = train_autoencoder_linear(X, hidden_layer_size=hidden_layer_size, epochs=600, lr=0.005)
_, _, _, o2 = forward_pass_l(X, W1, W2)
hidden_layer_output = W1 @ X.T
reconstructed_lae_data = o2 * 255


# Plot results of 2 dimensional embedding
plot_reconstructions_2(images_train.T, reconstructed_pca_data, reconstructed_lae_data, name='Reconstruction with 2 components')
plot_embeddings_2d(pca_data, hidden_layer_output, labels_train, name='Embedding')
