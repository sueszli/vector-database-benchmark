import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(X_train, y_train, X_test, y_test, clf, train_marker='o', test_marker='^', scoring='misclassification error', suppress_plot=False, print_model=True, title_fontsize=12, style='default', legend_loc='best'):
    if False:
        i = 10
        return i + 15
    "Plots learning curves of a classifier.\n\n    Parameters\n    ----------\n    X_train : array-like, shape = [n_samples, n_features]\n        Feature matrix of the training dataset.\n    y_train : array-like, shape = [n_samples]\n        True class labels of the training dataset.\n    X_test : array-like, shape = [n_samples, n_features]\n        Feature matrix of the test dataset.\n    y_test : array-like, shape = [n_samples]\n        True class labels of the test dataset.\n    clf : Classifier object. Must have a .predict .fit method.\n    train_marker : str (default: 'o')\n        Marker for the training set line plot.\n    test_marker : str (default: '^')\n        Marker for the test set line plot.\n    scoring : str (default: 'misclassification error')\n        If not 'misclassification error', accepts the following metrics\n        (from scikit-learn):\n        {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',\n        'f1_weighted', 'f1_samples', 'log_loss',\n        'precision', 'recall', 'roc_auc',\n        'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',\n        'median_absolute_error', 'r2'}\n    suppress_plot=False : bool (default: False)\n        Suppress matplotlib plots if True. Recommended\n        for testing purposes.\n    print_model : bool (default: True)\n        Print model parameters in plot title if True.\n    title_fontsize : int (default: 12)\n        Determines the size of the plot title font.\n    style : str (default: 'default')\n        Matplotlib style. For more styles, please see\n        https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html\n    legend_loc : str (default: 'best')\n        Where to place the plot legend:\n        {'best', 'upper left', 'upper right', 'lower left', 'lower right'}\n\n    Returns\n    ---------\n    errors : (training_error, test_error): tuple of lists\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/plotting/plot_learning_curves/\n\n    "
    if scoring != 'misclassification error':
        from sklearn import metrics
        scoring_func = {'accuracy': metrics.accuracy_score, 'average_precision': metrics.average_precision_score, 'f1': metrics.f1_score, 'f1_micro': metrics.f1_score, 'f1_macro': metrics.f1_score, 'f1_weighted': metrics.f1_score, 'f1_samples': metrics.f1_score, 'log_loss': metrics.log_loss, 'precision': metrics.precision_score, 'recall': metrics.recall_score, 'roc_auc': metrics.roc_auc_score, 'adjusted_rand_score': metrics.adjusted_rand_score, 'mean_absolute_error': metrics.mean_absolute_error, 'mean_squared_error': metrics.mean_squared_error, 'median_absolute_error': metrics.median_absolute_error, 'r2': metrics.r2_score}
        if scoring not in scoring_func.keys():
            raise AttributeError('scoring must be in', scoring_func.keys())
    else:

        def misclf_err(y_predict, y):
            if False:
                return 10
            return (y_predict != y).sum() / float(len(y))
        scoring_func = {'misclassification error': misclf_err}
    training_errors = []
    test_errors = []
    rng = [int(i) for i in np.linspace(0, X_train.shape[0], 11)][1:]
    for r in rng:
        model = clf.fit(X_train[:r], y_train[:r])
        y_train_predict = clf.predict(X_train[:r])
        y_test_predict = clf.predict(X_test)
        train_misclf = scoring_func[scoring](y_train[:r], y_train_predict)
        training_errors.append(train_misclf)
        test_misclf = scoring_func[scoring](y_test, y_test_predict)
        test_errors.append(test_misclf)
    if not suppress_plot:
        with plt.style.context(style):
            plt.plot(np.arange(10, 101, 10), training_errors, label='training set', marker=train_marker)
            plt.plot(np.arange(10, 101, 10), test_errors, label='test set', marker=test_marker)
            plt.xlabel('Training set size in percent')
    if not suppress_plot:
        with plt.style.context(style):
            plt.ylabel('Performance ({})'.format(scoring))
            if print_model:
                plt.title('Learning Curves\n\n{}\n'.format(model), fontsize=title_fontsize)
            plt.legend(loc=legend_loc, numpoints=1)
            plt.xlim([0, 110])
            max_y = max(max(test_errors), max(training_errors))
            min_y = min(min(test_errors), min(training_errors))
            plt.ylim([min_y - min_y * 0.15, max_y + max_y * 0.15])
    errors = (training_errors, test_errors)
    return errors