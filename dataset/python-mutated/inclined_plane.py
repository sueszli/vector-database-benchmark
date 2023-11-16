import argparse
import numpy as np
import torch
import pyro
from pyro.distributions import Normal, Uniform
from pyro.infer import EmpiricalMarginal, Importance
"\nSamantha really likes physics---but she likes Pyro even more. Instead of using\ncalculus to do her physics lab homework (which she could easily do), she's going\nto use bayesian inference. The problem setup is as follows. In lab she observed\na little box slide down an inclined plane (length of 2 meters and with an incline of\n30 degrees) 20 times. Each time she measured and recorded the descent time. The timing\ndevice she used has a known measurement error of 20 milliseconds. Using the observed\ndata, she wants to infer the coefficient of friction mu between the box and the inclined\nplane. She already has (deterministic) python code that can simulate the amount of time\nthat it takes the little box to slide down the inclined plane as a function of mu. Using\nPyro, she can reverse the simulator and infer mu from the observed descent times.\n"
little_g = 9.8
mu0 = 0.12
time_measurement_sigma = 0.02

def simulate(mu, length=2.0, phi=np.pi / 6.0, dt=0.005, noise_sigma=None):
    if False:
        return 10
    T = torch.zeros(())
    velocity = torch.zeros(())
    displacement = torch.zeros(())
    acceleration = torch.tensor(little_g * np.sin(phi)) - torch.tensor(little_g * np.cos(phi)) * mu
    if acceleration.numpy() <= 0.0:
        return torch.tensor(100000.0)
    while displacement.numpy() < length:
        displacement += velocity * dt
        velocity += acceleration * dt
        T += dt
    if noise_sigma is None:
        return T
    else:
        return T + noise_sigma * torch.randn(())

def analytic_T(mu, length=2.0, phi=np.pi / 6.0):
    if False:
        for i in range(10):
            print('nop')
    numerator = 2.0 * length
    denominator = little_g * (np.sin(phi) - mu * np.cos(phi))
    return np.sqrt(numerator / denominator)
print('generating simulated data using the true coefficient of friction %.3f' % mu0)
N_obs = 20
torch.manual_seed(2)
observed_data = torch.tensor([simulate(torch.tensor(mu0), noise_sigma=time_measurement_sigma) for _ in range(N_obs)])
observed_mean = np.mean([T.item() for T in observed_data])

def model(observed_data):
    if False:
        for i in range(10):
            print('nop')
    mu_prior = Uniform(0.0, 1.0)
    mu = pyro.sample('mu', mu_prior)

    def observe_T(T_obs, obs_name):
        if False:
            return 10
        T_simulated = simulate(mu)
        T_obs_dist = Normal(T_simulated, torch.tensor(time_measurement_sigma))
        pyro.sample(obs_name, T_obs_dist, obs=T_obs)
    for (i, T_obs) in enumerate(observed_data):
        observe_T(T_obs, 'obs_%d' % i)
    return mu

def main(args):
    if False:
        print('Hello World!')
    importance = Importance(model, guide=None, num_samples=args.num_samples)
    print('doing importance sampling...')
    emp_marginal = EmpiricalMarginal(importance.run(observed_data))
    posterior_mean = emp_marginal.mean
    posterior_std_dev = emp_marginal.variance.sqrt()
    inferred_mu = posterior_mean.item()
    inferred_mu_uncertainty = posterior_std_dev.item()
    print('the coefficient of friction inferred by pyro is %.3f +- %.3f' % (inferred_mu, inferred_mu_uncertainty))
    print('the mean observed descent time in the dataset is: %.4f seconds' % observed_mean)
    print('the (forward) simulated descent time for the inferred (mean) mu is: %.4f seconds' % simulate(posterior_mean).item())
    print(('disregarding measurement noise, elementary calculus gives the descent time\n' + 'for the inferred (mean) mu as: %.4f seconds') % analytic_T(posterior_mean.item()))
    '\n    ################## EXERCISE ###################\n    # vectorize the computations in this example! #\n    ###############################################\n    '
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='parse args')
    parser.add_argument('-n', '--num-samples', default=500, type=int)
    args = parser.parse_args()
    main(args)