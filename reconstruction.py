import numpy as np
from mnist import MNIST
from mnist_helper import mnist_downloader
from pca import pca
from plots import plot_loss, plot_reconstructions_1
from l_autoencoder import train_autoencoder_linear, forward_pass_l
from nl_autoencoder import train_autoencoder_nonlinear, forward_pass_nl
from loss_activation import error


# file executes pca with 32 dimension and trains a linear and nonlinear autoencoder with 32 hidden layer units
# plot the reconstruction and loss

# MNIST import
download_folder = r"mnist_helper/data"
mnist_downloader.download_and_unzip(download_folder)
mndata = MNIST(download_folder, return_type="numpy")
images_train, labels_train = mndata.load_training()
images_train = images_train[0:1000, :]
X = images_train/255


# PCA calculation
dimension = 32
pca_data, loading_matrix = pca(X.T, dimension=dimension, scree_plot=True)       # dimension x data_samples
reconstructed_pca_data = loading_matrix @ pca_data
pca_error = np.mean(error(X, reconstructed_pca_data), axis=0)
print(pca_error)
reconstructed_pca_data = reconstructed_pca_data * 255


# Train linear AE
hidden_layer_size = 32
W1, W2, error_history_1 = train_autoencoder_linear(X, hidden_layer_size=hidden_layer_size, epochs=600, lr=0.02)
_, _, _, o2 = forward_pass_l(X, W1, W2)
reconstructed_lae_data = o2 * 255


# Train nonlinear AE
W1, W2, error_history_2 = train_autoencoder_nonlinear(X, hidden_layer_size=hidden_layer_size, epochs=600, lr=0.1)
_, _, _, _, _, o2 = forward_pass_nl(X, W1, W2)
reconstructed_nlae_data = o2 * 255


# plot reconstruction and loss
plot_reconstructions_1(images_train.T, reconstructed_pca_data, reconstructed_lae_data, reconstructed_nlae_data, name='Reconstruction Plot')
plot_loss(error_history_1, error_history_2, pca_error)
