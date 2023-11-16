import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.utils import check_X_y

def visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, show_figure=True, save_figure=False):
    if False:
        i = 10
        return i + 15
    '\n    Utility function for visualizing the results in examples\n    Internal use only\n\n    :param clf_name: The name of the detector\n    :type clf_name: str\n\n    :param X_train: The training samples\n    :param X_train: numpy array of shape (n_samples, n_features)\n\n    :param y_train: The ground truth of training samples\n    :type y_train: list or array of shape (n_samples,)\n\n    :param X_test: The test samples\n    :type X_test: numpy array of shape (n_samples, n_features)\n\n    :param y_test: The ground truth of test samples\n    :type y_test: list or array of shape (n_samples,)\n\n    :param y_train_pred: The predicted outlier scores on the training samples\n    :type y_train_pred: numpy array of shape (n_samples, n_features)\n\n    :param y_test_pred: The predicted outlier scores on the test samples\n    :type y_test_pred: numpy array of shape (n_samples, n_features)\n\n    :param show_figure: If set to True, show the figure\n    :type show_figure: bool, optional (default=True)\n\n    :param save_figure: If set to True, save the figure to the local\n    :type save_figure: bool, optional (default=False)\n    '
    if X_train.shape[1] != 2 or X_test.shape[1] != 2:
        raise ValueError('Input data has to be 2-d for visualization. The input data has {shape}.'.format(shape=X_train.shape))
    (X_train, y_train) = check_X_y(X_train, y_train)
    (X_test, y_test) = check_X_y(X_test, y_test)
    c_train = get_color_codes(y_train)
    c_test = get_color_codes(y_test)
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle('Demo of {clf_name}'.format(clf_name=clf_name))
    fig.add_subplot(221)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=c_train)
    plt.title('Train ground truth')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='inlier', markerfacecolor='b', markersize=8), Line2D([0], [0], marker='^', color='w', label='outlier', markerfacecolor='r', markersize=8)]
    plt.legend(handles=legend_elements, loc=4)
    fig.add_subplot(222)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=c_test)
    plt.title('Test ground truth')
    plt.legend(handles=legend_elements, loc=4)
    fig.add_subplot(223)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred)
    plt.title('Train prediction by {clf_name}'.format(clf_name=clf_name))
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='inlier', markerfacecolor='0', markersize=8), Line2D([0], [0], marker='^', color='w', label='outlier', markerfacecolor='yellow', markersize=8)]
    plt.legend(handles=legend_elements, loc=4)
    fig.add_subplot(224)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred)
    plt.title('Test prediction by {clf_name}'.format(clf_name=clf_name))
    plt.legend(handles=legend_elements, loc=4)
    if save_figure:
        plt.savefig('{clf_name}.png'.format(clf_name=clf_name), dpi=300)
    if show_figure:
        plt.show()
    return

def get_color_codes(y):
    if False:
        for i in range(10):
            print('nop')
    'Internal function to generate color codes for inliers and outliers.\n    Inliers (0): blue; Outlier (1): red.\n\n    Parameters\n    ----------\n    y : list or numpy array of shape (n_samples,)\n        The ground truth. Binary (0: inliers, 1: outliers).\n\n    Returns\n    -------\n    c : numpy array of shape (n_samples,)\n        Color codes.\n\n    '
    y = column_or_1d(y)
    c = np.full([len(y)], 'b', dtype=str)
    outliers_ind = np.where(y == 1)
    c[outliers_ind] = 'r'
    return c