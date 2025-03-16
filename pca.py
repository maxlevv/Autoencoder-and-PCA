import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


def get_loading_matrix(X, dimension, scree_plot:bool = True):
    covariance_matrix = np.cov(X, rowvar=True)
    eig_values, eig_vectors = eigh(covariance_matrix)  # already normalized eigenvectors
    sort_index = np.argsort(eig_values)
    eig_values_sorted = eig_values[sort_index[::-1]]
    eig_vectors_sorted = eig_vectors[:, sort_index[::-1]]
    loading_matrix = eig_vectors_sorted

    percentage = (eig_values_sorted / eig_values_sorted.sum() * 100).round(4)  # percentage of influence of each principle component
    cum_percentage = np.cumsum(percentage)  # adds to 100
    print(np.cumsum(percentage[:dimension])[dimension-1])

    if scree_plot == True:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.bar(x=range(1, len(eig_values_sorted) + 1 - 684),
                height=percentage[:100], width=0.8)  # , tick_label=labels)
        ax.set_xlabel('Principal Component', fontsize=20)
        ax.set_ylabel('Percentage', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid()
        fig.savefig('Scree Plot.png')
        fig.show()

    return loading_matrix


def pca(X, dimension:int, scree_plot:bool = True):
    loading_matrix = get_loading_matrix(X, dimension, scree_plot)
    loading_matrix = loading_matrix[:, :dimension]
    pca_data = loading_matrix.T @ X
    return pca_data, loading_matrix
