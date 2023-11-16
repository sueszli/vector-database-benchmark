"""
==============================================
Lasso model selection via information criteria
==============================================

This example reproduces the example of Fig. 2 of [ZHT2007]_. A
:class:`~sklearn.linear_model.LassoLarsIC` estimator is fit on a
diabetes dataset and the AIC and the BIC criteria are used to select
the best model.

.. note::
    It is important to note that the optimization to find `alpha` with
    :class:`~sklearn.linear_model.LassoLarsIC` relies on the AIC or BIC
    criteria that are computed in-sample, thus on the training set directly.
    This approach differs from the cross-validation procedure. For a comparison
    of the two approaches, you can refer to the following example:
    :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py`.

.. topic:: References

    .. [ZHT2007] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
       "On the degrees of freedom of the lasso."
       The Annals of Statistics 35.5 (2007): 2173-2192.
       <0712.0881>`
"""
from sklearn.datasets import load_diabetes
(X, y) = load_diabetes(return_X_y=True, as_frame=True)
n_samples = X.shape[0]
X.head()
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion='aic')).fit(X, y)

def zou_et_al_criterion_rescaling(criterion, n_samples, noise_variance):
    if False:
        while True:
            i = 10
    'Rescale the information criterion to follow the definition of Zou et al.'
    return criterion - n_samples * np.log(2 * np.pi * noise_variance) - n_samples
import numpy as np
aic_criterion = zou_et_al_criterion_rescaling(lasso_lars_ic[-1].criterion_, n_samples, lasso_lars_ic[-1].noise_variance_)
index_alpha_path_aic = np.flatnonzero(lasso_lars_ic[-1].alphas_ == lasso_lars_ic[-1].alpha_)[0]
lasso_lars_ic.set_params(lassolarsic__criterion='bic').fit(X, y)
bic_criterion = zou_et_al_criterion_rescaling(lasso_lars_ic[-1].criterion_, n_samples, lasso_lars_ic[-1].noise_variance_)
index_alpha_path_bic = np.flatnonzero(lasso_lars_ic[-1].alphas_ == lasso_lars_ic[-1].alpha_)[0]
index_alpha_path_aic == index_alpha_path_bic
import matplotlib.pyplot as plt
plt.plot(aic_criterion, color='tab:blue', marker='o', label='AIC criterion')
plt.plot(bic_criterion, color='tab:orange', marker='o', label='BIC criterion')
plt.vlines(index_alpha_path_bic, aic_criterion.min(), aic_criterion.max(), color='black', linestyle='--', label='Selected alpha')
plt.legend()
plt.ylabel('Information criterion')
plt.xlabel('Lasso model sequence')
_ = plt.title('Lasso model selection via AIC and BIC')