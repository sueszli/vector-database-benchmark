"""
=========================================================
Plot classification boundaries with different SVM Kernels
=========================================================
This example shows how different kernels in a :class:`~sklearn.svm.SVC` (Support Vector
Classifier) influence the classification boundaries in a binary, two-dimensional
classification problem.

SVCs aim to find a hyperplane that effectively separates the classes in their training
data by maximizing the margin between the outermost data points of each class. This is
achieved by finding the best weight vector :math:`w` that defines the decision boundary
hyperplane and minimizes the sum of hinge losses for misclassified samples, as measured
by the :func:`~sklearn.metrics.hinge_loss` function. By default, regularization is
applied with the parameter `C=1`, which allows for a certain degree of misclassification
tolerance.

If the data is not linearly separable in the original feature space, a non-linear kernel
parameter can be set. Depending on the kernel, the process involves adding new features
or transforming existing features to enrich and potentially add meaning to the data.
When a kernel other than `"linear"` is set, the SVC applies the `kernel trick
<https://en.wikipedia.org/wiki/Kernel_method#Mathematics:_the_kernel_trick>`__, which
computes the similarity between pairs of data points using the kernel function without
explicitly transforming the entire dataset. The kernel trick surpasses the otherwise
necessary matrix transformation of the whole dataset by only considering the relations
between all pairs of data points. The kernel function maps two vectors (each pair of
observations) to their similarity using their dot product.

The hyperplane can then be calculated using the kernel function as if the dataset were
represented in a higher-dimensional space. Using a kernel function instead of an
explicit matrix transformation improves performance, as the kernel function has a time
complexity of :math:`O({n}^2)`, whereas matrix transformation scales according to the
specific transformation being applied.

In this example, we compare the most common kernel types of Support Vector Machines: the
linear kernel (`"linear"`), the polynomial kernel (`"poly"`), the radial basis function
kernel (`"rbf"`) and the sigmoid kernel (`"sigmoid"`).
"""
import matplotlib.pyplot as plt
import numpy as np
X = np.array([[0.4, -0.7], [-1.5, -1.0], [-1.4, -0.9], [-1.3, -1.2], [-1.1, -0.2], [-1.2, -0.4], [-0.5, 1.2], [-1.5, 2.1], [1.0, 1.0], [1.3, 0.8], [1.2, 0.5], [0.2, -2.0], [0.5, -2.4], [0.2, -2.3], [0.0, -2.7], [1.3, 2.1]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
(fig, ax) = plt.subplots(figsize=(4, 3))
(x_min, x_max, y_min, y_max) = (-3, 3, -3, 3)
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
scatter = ax.scatter(X[:, 0], X[:, 1], s=150, c=y, label=y, edgecolors='k')
ax.legend(*scatter.legend_elements(), loc='upper right', title='Classes')
ax.set_title('Samples in two-dimensional feature space')
_ = plt.show()
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

def plot_training_data_with_decision_boundary(kernel):
    if False:
        for i in range(10):
            print('nop')
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)
    (_, ax) = plt.subplots(figsize=(4, 3))
    (x_min, x_max, y_min, y_max) = (-3, 3, -3, 3)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    common_params = {'estimator': clf, 'X': X, 'ax': ax}
    DecisionBoundaryDisplay.from_estimator(**common_params, response_method='predict', plot_method='pcolormesh', alpha=0.3)
    DecisionBoundaryDisplay.from_estimator(**common_params, response_method='decision_function', plot_method='contour', levels=[-1, 0, 1], colors=['k', 'k', 'k'], linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=250, facecolors='none', edgecolors='k')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors='k')
    ax.legend(*scatter.legend_elements(), loc='upper right', title='Classes')
    ax.set_title(f' Decision boundaries of {kernel} kernel in SVC')
    _ = plt.show()
plot_training_data_with_decision_boundary('linear')
plot_training_data_with_decision_boundary('poly')
plot_training_data_with_decision_boundary('rbf')
plot_training_data_with_decision_boundary('sigmoid')