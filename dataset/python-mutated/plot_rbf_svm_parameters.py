import matplotlib.pyplot as plt
from sklearn.svm import SVC
from .plot_2d_separator import plot_2d_separator
from .tools import make_handcrafted_dataset
from .plot_helpers import discrete_scatter

def plot_svm(log_C, log_gamma, ax=None):
    if False:
        i = 10
        return i + 15
    (X, y) = make_handcrafted_dataset()
    C = 10.0 ** log_C
    gamma = 10.0 ** log_gamma
    svm = SVC(kernel='rbf', C=C, gamma=gamma).fit(X, y)
    if ax is None:
        ax = plt.gca()
    plot_2d_separator(svm, X, ax=ax, eps=0.5)
    discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    sv = svm.support_vectors_
    sv_labels = svm.dual_coef_.ravel() > 0
    discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3, ax=ax)
    ax.set_title('C = %.4f gamma = %.4f' % (C, gamma))

def plot_svm_interactive():
    if False:
        return 10
    from IPython.html.widgets import interactive, FloatSlider
    C_slider = FloatSlider(min=-3, max=3, step=0.1, value=0, readout=False)
    gamma_slider = FloatSlider(min=-2, max=2, step=0.1, value=0, readout=False)
    return interactive(plot_svm, log_C=C_slider, log_gamma=gamma_slider)