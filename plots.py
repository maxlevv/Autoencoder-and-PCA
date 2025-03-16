import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# file contains all functions to plot the results


def plot_loss(error_history_linear, error_history_nonlinear, pca_error):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(error_history_linear[5:], label='loss linear AE')
    ax.plot(error_history_nonlinear[5:], label='loss nonlinear AE')
    ax.hlines(y=pca_error, xmin=0, xmax=len(error_history_linear) - 5, color='r', label='loss PCA')
    ax.set_xlabel('epochs', fontsize=20)
    ax.legend(loc='upper right', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()
    fig.savefig('Loss Plot.png')
    fig.show()


def plot_clusters_3d(data, labels_train, name):
    n = data.shape[1]
    data = np.array(data, dtype='float')
    x = data[0, :]
    y = data[1, :]
    z = data[2, :]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title(name)
    ax.axis('off')
    scatter = ax.scatter(x, y, c=labels_train[:n], cmap='tab10', marker='o', s=12)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    #plt.savefig(name + '.png')
    plt.show()


def plot_clusters_2d(data, labels_train, name):
    n = data.shape[1]
    data = np.array(data, dtype='float')
    x = data[0, :]
    y = data[1, :]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title(name)
    ax.axis('off')
    scatter = ax.scatter(x, y, c=labels_train[:n], cmap='tab10', marker='o', s=12)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.show()


def plot_images(original, reconstructed_1,  name):
    fig, axs = plt.subplots(2, 4, figsize=(12, 4))
    fig.suptitle(name)
    fig.subplots_adjust()
    for i in range(2):
        for j in range(2):
            image = original[:, 2*i+j]
            image = np.array(image, dtype='float')
            image = image.reshape((28, 28))
            axs[i, j].imshow(image, cmap='gray')
    for i in range(2):
        for j in range(2):
            image = reconstructed_1[:, 2*i+j]
            image = np.array(image, dtype='float')
            image = image.reshape((28, 28))
            axs[i, 2+j].imshow(image, cmap='gray')
    plt.show()


def plot_reconstructions_1(original, reconstructed1, reconstructed2, reconstructed3, name):
    fig = plt.figure(figsize=(16, 4))
    outer = gridspec.GridSpec(1, 4, wspace=0.2, hspace=0.2)
    list = [original, reconstructed1, reconstructed2, reconstructed3]
    titles = ['original', 'PCA', 'Linear AE', 'Nonlinear AE']
    for i in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                 subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        ax = plt.Subplot(fig, outer[i])
        ax.set_title(titles[i], fontsize=20)
        ax.axis('off')
        fig.add_subplot(ax)
        images = list[i]

        for j in range(4):
            image = images[:, j]
            image = np.array(image, dtype='float')
            image = image.reshape((28, 28))
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(image, cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)
    fig.savefig(name + '.png')
    fig.show()


def plot_reconstructions_2(original, reconstructed1, reconstructed2, name):
    fig = plt.figure(figsize=(12, 4))
    outer = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.4)
    list = [original, reconstructed1, reconstructed2]
    titles = ['original', 'PCA reconstructed', 'Linear AE reconstructed']
    for i in range(3):
        inner = gridspec.GridSpecFromSubplotSpec(1, 10,
                                                 subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        ax = plt.Subplot(fig, outer[i])
        ax.set_title(titles[i])
        ax.axis('off')
        fig.add_subplot(ax)
        images = list[i]

        for j in range(10):
            mnist_numbers = [1, 3, 5, 7, 2, 0, 13, 15, 17, 4]
            k = mnist_numbers[j]
            image = images[:, k]
            image = np.array(image, dtype='float')
            image = image.reshape((28, 28))
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(image, cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)
    fig.savefig(name + '.png')
    fig.show()

def plot_embeddings_2d(data_1, data_2, labels_train, name):
    n = data_1.shape[1]
    data_1 = np.array(data_1, dtype='float')
    x1 = data_1[0, :]
    y1 = data_1[1, :]

    data_2 = np.array(data_2, dtype='float')
    x2 = data_2[0, :]
    y2 = data_2[1, :]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3)
    scatter_1 = ax[0].scatter(x1, y1, c=labels_train[:n], cmap='tab10', marker='o', s=12)
    scatter_2 = ax[1].scatter(x2, y2, c=labels_train[:n], cmap='tab10', marker='o', s=12)
    legend1 = ax[0].legend(*scatter_1.legend_elements(), loc="upper left", title="Classes")
    ax[0].add_artist(legend1)
    ax[0].set_title('PCA embedding')
    ax[1].set_title('Linear AE embedding')
    ax[0].axis('off')
    ax[1].axis('off')
    fig.savefig(name + '.png')
    plt.show()
