"""
================================
Gaussian Mixture Model Selection
================================

This example shows that model selection can be performed with Gaussian Mixture
Models (GMM) using :ref:`information-theory criteria <aic_bic>`. Model selection
concerns both the covariance type and the number of components in the model.

In this case, both the Akaike Information Criterion (AIC) and the Bayes
Information Criterion (BIC) provide the right result, but we only demo the
latter as BIC is better suited to identify the true model among a set of
candidates. Unlike Bayesian procedures, such inferences are prior-free.

"""
import numpy as np
n_samples = 500
np.random.seed(0)
C = np.array([[0.0, -0.1], [1.7, 0.4]])
component_1 = np.dot(np.random.randn(n_samples, 2), C)
component_2 = 0.7 * np.random.randn(n_samples, 2) + np.array([-4, 1])
X = np.concatenate([component_1, component_2])
import matplotlib.pyplot as plt
plt.scatter(component_1[:, 0], component_1[:, 1], s=0.8)
plt.scatter(component_2[:, 0], component_2[:, 1], s=0.8)
plt.title('Gaussian Mixture components')
plt.axis('equal')
plt.show()
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

def gmm_bic_score(estimator, X):
    if False:
        i = 10
        return i + 15
    'Callable to pass to GridSearchCV that will use the BIC score.'
    return -estimator.bic(X)
param_grid = {'n_components': range(1, 7), 'covariance_type': ['spherical', 'tied', 'diag', 'full']}
grid_search = GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score)
grid_search.fit(X)
import pandas as pd
df = pd.DataFrame(grid_search.cv_results_)[['param_n_components', 'param_covariance_type', 'mean_test_score']]
df['mean_test_score'] = -df['mean_test_score']
df = df.rename(columns={'param_n_components': 'Number of components', 'param_covariance_type': 'Type of covariance', 'mean_test_score': 'BIC score'})
df.sort_values(by='BIC score').head()
import seaborn as sns
sns.catplot(data=df, kind='bar', x='Number of components', y='BIC score', hue='Type of covariance')
plt.show()
from matplotlib.patches import Ellipse
from scipy import linalg
color_iter = sns.color_palette('tab10', 2)[::-1]
Y_ = grid_search.predict(X)
(fig, ax) = plt.subplots()
for (i, (mean, cov, color)) in enumerate(zip(grid_search.best_estimator_.means_, grid_search.best_estimator_.covariances_, color_iter)):
    (v, w) = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ellipse = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
    ellipse.set_clip_box(fig.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)
plt.title(f"Selected GMM: {grid_search.best_params_['covariance_type']} model, {grid_search.best_params_['n_components']} components")
plt.axis('equal')
plt.show()