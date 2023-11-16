"""
================================
Image denoising using kernel PCA
================================

This example shows how to use :class:`~sklearn.decomposition.KernelPCA` to
denoise images. In short, we take advantage of the approximation function
learned during `fit` to reconstruct the original image.

We will compare the results with an exact reconstruction using
:class:`~sklearn.decomposition.PCA`.

We will use USPS digits dataset to reproduce presented in Sect. 4 of [1]_.

.. topic:: References

   .. [1] `Bakır, Gökhan H., Jason Weston, and Bernhard Schölkopf.
      "Learning to find pre-images."
      Advances in neural information processing systems 16 (2004): 449-456.
      <https://papers.nips.cc/paper/2003/file/ac1ad983e08ad3304a97e147f522747e-Paper.pdf>`_

"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
(X, y) = fetch_openml(data_id=41082, as_frame=False, return_X_y=True, parser='pandas')
X = MinMaxScaler().fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, random_state=0, train_size=1000, test_size=100)
rng = np.random.RandomState(0)
noise = rng.normal(scale=0.25, size=X_test.shape)
X_test_noisy = X_test + noise
noise = rng.normal(scale=0.25, size=X_train.shape)
X_train_noisy = X_train + noise
import matplotlib.pyplot as plt

def plot_digits(X, title):
    if False:
        return 10
    'Small helper function to plot 100 digits.'
    (fig, axs) = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for (img, ax) in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap='Greys')
        ax.axis('off')
    fig.suptitle(title, fontsize=24)
plot_digits(X_test, 'Uncorrupted test images')
plot_digits(X_test_noisy, f'Noisy test images\nMSE: {np.mean((X_test - X_test_noisy) ** 2):.2f}')
from sklearn.decomposition import PCA, KernelPCA
pca = PCA(n_components=32, random_state=42)
kernel_pca = KernelPCA(n_components=400, kernel='rbf', gamma=0.001, fit_inverse_transform=True, alpha=0.005, random_state=42)
pca.fit(X_train_noisy)
_ = kernel_pca.fit(X_train_noisy)
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(kernel_pca.transform(X_test_noisy))
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_noisy))
plot_digits(X_test, 'Uncorrupted test images')
plot_digits(X_reconstructed_pca, f'PCA reconstruction\nMSE: {np.mean((X_test - X_reconstructed_pca) ** 2):.2f}')
plot_digits(X_reconstructed_kernel_pca, f'Kernel PCA reconstruction\nMSE: {np.mean((X_test - X_reconstructed_kernel_pca) ** 2):.2f}')