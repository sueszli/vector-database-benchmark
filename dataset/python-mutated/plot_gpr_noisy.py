"""
=========================================================================
Ability of Gaussian process regression (GPR) to estimate data noise-level
=========================================================================

This example shows the ability of the
:class:`~sklearn.gaussian_process.kernels.WhiteKernel` to estimate the noise
level in the data. Moreover, we show the importance of kernel hyperparameters
initialization.
"""
import numpy as np

def target_generator(X, add_noise=False):
    if False:
        i = 10
        return i + 15
    target = 0.5 + np.sin(3 * X)
    if add_noise:
        rng = np.random.RandomState(1)
        target += rng.normal(0, 0.3, size=target.shape)
    return target.squeeze()
X = np.linspace(0, 5, num=30).reshape(-1, 1)
y = target_generator(X, add_noise=False)
import matplotlib.pyplot as plt
plt.plot(X, y, label='Expected signal')
plt.legend()
plt.xlabel('X')
_ = plt.ylabel('y')
rng = np.random.RandomState(0)
X_train = rng.uniform(0, 5, size=20).reshape(-1, 1)
y_train = target_generator(X_train, add_noise=True)
plt.plot(X, y, label='Expected signal')
plt.scatter(x=X_train[:, 0], y=y_train, color='black', alpha=0.4, label='Observations')
plt.legend()
plt.xlabel('X')
_ = plt.ylabel('y')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
kernel = 1.0 * RBF(length_scale=10.0, length_scale_bounds=(0.01, 1000.0)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-05, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
gpr.fit(X_train, y_train)
(y_mean, y_std) = gpr.predict(X, return_std=True)
plt.plot(X, y, label='Expected signal')
plt.scatter(x=X_train[:, 0], y=y_train, color='black', alpha=0.4, label='Observations')
plt.errorbar(X, y_mean, y_std)
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
_ = plt.title(f'Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta)}', fontsize=8)
kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(0.01, 1000.0)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
gpr.fit(X_train, y_train)
(y_mean, y_std) = gpr.predict(X, return_std=True)
plt.plot(X, y, label='Expected signal')
plt.scatter(x=X_train[:, 0], y=y_train, color='black', alpha=0.4, label='Observations')
plt.errorbar(X, y_mean, y_std)
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
_ = plt.title(f'Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta)}', fontsize=8)
from matplotlib.colors import LogNorm
length_scale = np.logspace(-2, 4, num=50)
noise_level = np.logspace(-2, 1, num=50)
(length_scale_grid, noise_level_grid) = np.meshgrid(length_scale, noise_level)
log_marginal_likelihood = [gpr.log_marginal_likelihood(theta=np.log([0.36, scale, noise])) for (scale, noise) in zip(length_scale_grid.ravel(), noise_level_grid.ravel())]
log_marginal_likelihood = np.reshape(log_marginal_likelihood, newshape=noise_level_grid.shape)
(vmin, vmax) = ((-log_marginal_likelihood).min(), 50)
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=50), decimals=1)
plt.contour(length_scale_grid, noise_level_grid, -log_marginal_likelihood, levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Length-scale')
plt.ylabel('Noise-level')
plt.title('Log-marginal-likelihood')
plt.show()