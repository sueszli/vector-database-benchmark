"""
==========================================================================
Illustration of prior and posterior Gaussian process for different kernels
==========================================================================

This example illustrates the prior and posterior of a
:class:`~sklearn.gaussian_process.GaussianProcessRegressor` with different
kernels. Mean, standard deviation, and 5 samples are shown for both prior
and posterior distributions.

Here, we only give some illustration. To know more about kernels' formulation,
refer to the :ref:`User Guide <gp_kernels>`.

"""
import matplotlib.pyplot as plt
import numpy as np

def plot_gpr_samples(gpr_model, n_samples, ax):
    if False:
        return 10
    'Plot samples drawn from the Gaussian process model.\n\n    If the Gaussian process model is not trained then the drawn samples are\n    drawn from the prior distribution. Otherwise, the samples are drawn from\n    the posterior distribution. Be aware that a sample here corresponds to a\n    function.\n\n    Parameters\n    ----------\n    gpr_model : `GaussianProcessRegressor`\n        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.\n    n_samples : int\n        The number of samples to draw from the Gaussian process distribution.\n    ax : matplotlib axis\n        The matplotlib axis where to plot the samples.\n    '
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)
    (y_mean, y_std) = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)
    for (idx, single_prior) in enumerate(y_samples.T):
        ax.plot(x, single_prior, linestyle='--', alpha=0.7, label=f'Sampled function #{idx + 1}')
    ax.plot(x, y_mean, color='black', label='Mean')
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1, color='black', label='$\\pm$ 1 std. dev.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([-3, 3])
rng = np.random.RandomState(4)
X_train = rng.uniform(0, 5, 10).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
n_samples = 5
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
(fig, axs) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title('Samples from prior distribution')
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
axs[1].set_title('Samples from posterior distribution')
fig.suptitle('Radial Basis Function kernel', fontsize=18)
plt.tight_layout()
print(f'Kernel parameters before fit:\n{kernel})')
print(f'Kernel parameters after fit: \n{gpr.kernel_} \nLog-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}')
from sklearn.gaussian_process.kernels import RationalQuadratic
kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-05, 1000000000000000.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
(fig, axs) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title('Samples from prior distribution')
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
axs[1].set_title('Samples from posterior distribution')
fig.suptitle('Rational Quadratic kernel', fontsize=18)
plt.tight_layout()
print(f'Kernel parameters before fit:\n{kernel})')
print(f'Kernel parameters after fit: \n{gpr.kernel_} \nLog-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}')
from sklearn.gaussian_process.kernels import ExpSineSquared
kernel = 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
(fig, axs) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title('Samples from prior distribution')
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
axs[1].set_title('Samples from posterior distribution')
fig.suptitle('Exp-Sine-Squared kernel', fontsize=18)
plt.tight_layout()
print(f'Kernel parameters before fit:\n{kernel})')
print(f'Kernel parameters after fit: \n{gpr.kernel_} \nLog-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}')
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct
kernel = ConstantKernel(0.1, (0.01, 10.0)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
(fig, axs) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title('Samples from prior distribution')
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
axs[1].set_title('Samples from posterior distribution')
fig.suptitle('Dot-product kernel', fontsize=18)
plt.tight_layout()
print(f'Kernel parameters before fit:\n{kernel})')
print(f'Kernel parameters after fit: \n{gpr.kernel_} \nLog-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}')
from sklearn.gaussian_process.kernels import Matern
kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
(fig, axs) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title('Samples from prior distribution')
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
axs[1].set_title('Samples from posterior distribution')
fig.suptitle('Matérn kernel', fontsize=18)
plt.tight_layout()
print(f'Kernel parameters before fit:\n{kernel})')
print(f'Kernel parameters after fit: \n{gpr.kernel_} \nLog-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}')