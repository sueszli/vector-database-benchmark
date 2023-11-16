"""
=============================================================
Compare the effect of different scalers on data with outliers
=============================================================

Feature 0 (median income in a block) and feature 5 (average house occupancy) of
the :ref:`california_housing_dataset` have very
different scales and contain some very large outliers. These two
characteristics lead to difficulties to visualize the data and, more
importantly, they can degrade the predictive performance of many machine
learning algorithms. Unscaled data can also slow down or even prevent the
convergence of many gradient-based estimators.

Indeed many estimators are designed with the assumption that each feature takes
values close to zero or more importantly that all features vary on comparable
scales. In particular, metric-based and gradient-based estimators often assume
approximately standardized data (centered features with unit variances). A
notable exception are decision tree-based estimators that are robust to
arbitrary scaling of the data.

This example uses different scalers, transformers, and normalizers to bring the
data within a pre-defined range.

Scalers are linear (or more precisely affine) transformers and differ from each
other in the way they estimate the parameters used to shift and scale each
feature.

:class:`~sklearn.preprocessing.QuantileTransformer` provides non-linear
transformations in which distances
between marginal outliers and inliers are shrunk.
:class:`~sklearn.preprocessing.PowerTransformer` provides
non-linear transformations in which data is mapped to a normal distribution to
stabilize variance and minimize skewness.

Unlike the previous transformations, normalization refers to a per sample
transformation instead of a per feature transformation.

The following code is a bit verbose, feel free to jump directly to the analysis
of the results_.

"""
import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler, minmax_scale
dataset = fetch_california_housing()
(X_full, y_full) = (dataset.data, dataset.target)
feature_names = dataset.feature_names
feature_mapping = {'MedInc': 'Median income in block', 'HouseAge': 'Median house age in block', 'AveRooms': 'Average number of rooms', 'AveBedrms': 'Average number of bedrooms', 'Population': 'Block population', 'AveOccup': 'Average house occupancy', 'Latitude': 'House block latitude', 'Longitude': 'House block longitude'}
features = ['MedInc', 'AveOccup']
features_idx = [feature_names.index(feature) for feature in features]
X = X_full[:, features_idx]
distributions = [('Unscaled data', X), ('Data after standard scaling', StandardScaler().fit_transform(X)), ('Data after min-max scaling', MinMaxScaler().fit_transform(X)), ('Data after max-abs scaling', MaxAbsScaler().fit_transform(X)), ('Data after robust scaling', RobustScaler(quantile_range=(25, 75)).fit_transform(X)), ('Data after power transformation (Yeo-Johnson)', PowerTransformer(method='yeo-johnson').fit_transform(X)), ('Data after power transformation (Box-Cox)', PowerTransformer(method='box-cox').fit_transform(X)), ('Data after quantile transformation (uniform pdf)', QuantileTransformer(output_distribution='uniform', random_state=42).fit_transform(X)), ('Data after quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal', random_state=42).fit_transform(X)), ('Data after sample-wise L2 normalizing', Normalizer().fit_transform(X))]
y = minmax_scale(y_full)
cmap = getattr(cm, 'plasma_r', cm.hot_r)

def create_axes(title, figsize=(16, 6)):
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    (left, width) = (0.1, 0.22)
    (bottom, height) = (0.1, 0.7)
    bottom_h = height + 0.15
    left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]
    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)
    left = width + left + 0.2
    left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]
    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)
    (left, width) = (width + left + 0.13, 0.01)
    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)
    return ((ax_scatter, ax_histy, ax_histx), (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom), ax_colorbar)

def plot_distribution(axes, X, y, hist_nbins=50, title='', x0_label='', x1_label=''):
    if False:
        while True:
            i = 10
    (ax, hist_X1, hist_X0) = axes
    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)
    colors = cmap(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0, c=colors)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal', color='grey', ec='grey')
    hist_X1.axis('off')
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical', color='grey', ec='grey')
    hist_X0.axis('off')

def make_plot(item_idx):
    if False:
        i = 10
        return i + 15
    (title, X) = distributions[item_idx]
    (ax_zoom_out, ax_zoom_in, ax_colorbar) = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(axarr[0], X, y, hist_nbins=200, x0_label=feature_mapping[features[0]], x1_label=feature_mapping[features[1]], title='Full data')
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)
    non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1)
    plot_distribution(axarr[1], X[non_outliers_mask], y[non_outliers_mask], hist_nbins=50, x0_label=feature_mapping[features[0]], x1_label=feature_mapping[features[1]], title='Zoom-in')
    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, orientation='vertical', label='Color mapping for values of y')
make_plot(0)
make_plot(1)
make_plot(2)
make_plot(3)
make_plot(4)
make_plot(5)
make_plot(6)
make_plot(7)
make_plot(8)
make_plot(9)
plt.show()