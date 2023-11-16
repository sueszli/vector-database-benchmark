import argparse
import logging
import math
import re
from collections import OrderedDict
import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, config_enumerate, infer_discrete
from pyro.infer.autoguide import init_to_value
from pyro.ops.special import safe_log
from pyro.ops.tensor_utils import convolve
from pyro.util import warn_if_nan
logging.basicConfig(format='%(message)s', level=logging.INFO)

def global_model(population):
    if False:
        while True:
            i = 10
    tau = args.recovery_time
    R0 = pyro.sample('R0', dist.LogNormal(0.0, 1.0))
    rho = pyro.sample('rho', dist.Uniform(0, 1))
    rate_s = -R0 / (tau * population)
    prob_i = 1 / (1 + tau)
    return (rate_s, prob_i, rho)

def discrete_model(args, data):
    if False:
        for i in range(10):
            print('nop')
    (rate_s, prob_i, rho) = global_model(args.population)
    S = torch.tensor(args.population - 1.0)
    I = torch.tensor(1.0)
    for (t, datum) in enumerate(data):
        S2I = pyro.sample('S2I_{}'.format(t), dist.Binomial(S, -(rate_s * I).expm1()))
        I2R = pyro.sample('I2R_{}'.format(t), dist.Binomial(I, prob_i))
        S = pyro.deterministic('S_{}'.format(t), S - S2I)
        I = pyro.deterministic('I_{}'.format(t), I + S2I - I2R)
        pyro.sample('obs_{}'.format(t), dist.ExtendedBinomial(S2I, rho), obs=datum)

def generate_data(args):
    if False:
        print('Hello World!')
    logging.info('Generating data...')
    params = {'R0': torch.tensor(args.basic_reproduction_number), 'rho': torch.tensor(args.response_rate)}
    empty_data = [None] * (args.duration + args.forecast)
    for attempt in range(100):
        with poutine.trace() as tr:
            with poutine.condition(data=params):
                discrete_model(args, empty_data)
        obs = torch.stack([site['value'] for (name, site) in tr.trace.nodes.items() if re.match('obs_[0-9]+', name)])
        S2I = torch.stack([site['value'] for (name, site) in tr.trace.nodes.items() if re.match('S2I_[0-9]+', name)])
        assert len(obs) == len(empty_data)
        obs_sum = int(obs[:args.duration].sum())
        S2I_sum = int(S2I[:args.duration].sum())
        if obs_sum >= args.min_observations:
            logging.info('Observed {:d}/{:d} infections:\n{}'.format(obs_sum, S2I_sum, ' '.join([str(int(x)) for x in obs[:args.duration]])))
            return {'S2I': S2I, 'obs': obs}
    raise ValueError('Failed to generate {} observations. Try increasing --population or decreasing --min-observations'.format(args.min_observations))

@config_enumerate
def reparameterized_discrete_model(args, data):
    if False:
        i = 10
        return i + 15
    (rate_s, prob_i, rho) = global_model(args.population)
    S_curr = torch.tensor(args.population - 1.0)
    I_curr = torch.tensor(1.0)
    for (t, datum) in enumerate(data):
        (S_prev, I_prev) = (S_curr, I_curr)
        S_curr = pyro.sample('S_{}'.format(t), dist.Binomial(args.population, 0.5).mask(False))
        I_curr = pyro.sample('I_{}'.format(t), dist.Binomial(args.population, 0.5).mask(False))
        S2I = S_prev - S_curr
        I2R = I_prev - I_curr + S2I
        pyro.sample('S2I_{}'.format(t), dist.ExtendedBinomial(S_prev, -(rate_s * I_prev).expm1()), obs=S2I)
        pyro.sample('I2R_{}'.format(t), dist.ExtendedBinomial(I_prev, prob_i), obs=I2R)
        pyro.sample('obs_{}'.format(t), dist.ExtendedBinomial(S2I, rho), obs=datum)

def infer_hmc_enum(args, data):
    if False:
        for i in range(10):
            print('nop')
    model = reparameterized_discrete_model
    return _infer_hmc(args, data, model)

def _infer_hmc(args, data, model, init_values={}):
    if False:
        return 10
    logging.info('Running inference...')
    kernel = NUTS(model, full_mass=[('R0', 'rho')], max_tree_depth=args.max_tree_depth, init_strategy=init_to_value(values=init_values), jit_compile=args.jit, ignore_jit_warnings=True)
    energies = []

    def hook_fn(kernel, *unused):
        if False:
            return 10
        e = float(kernel._potential_energy_last)
        energies.append(e)
        if args.verbose:
            logging.info('potential = {:0.6g}'.format(e))
    mcmc = MCMC(kernel, hook_fn=hook_fn, num_samples=args.num_samples, warmup_steps=args.warmup_steps)
    mcmc.run(args, data)
    mcmc.summary()
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(energies)
        plt.xlabel('MCMC step')
        plt.ylabel('potential energy')
        plt.title('MCMC energy trace')
        plt.tight_layout()
    samples = mcmc.get_samples()
    return samples

def quantize(name, x_real, min, max):
    if False:
        print('Hello World!')
    '\n    Randomly quantize in a way that preserves probability mass.\n    We use a piecewise polynomial spline of order 3.\n    '
    assert min < max
    lb = x_real.detach().floor()
    s = x_real - lb
    ss = s * s
    t = 1 - s
    tt = t * t
    probs = torch.stack([t * tt, 4 + ss * (3 * s - 6), 4 + tt * (3 * t - 6), s * ss], dim=-1) * (1 / 6)
    q = pyro.sample('Q_' + name, dist.Categorical(probs)).type_as(x_real)
    x = lb + q - 1
    x = torch.max(x, 2 * min - 1 - x)
    x = torch.min(x, 2 * max + 1 - x)
    return pyro.deterministic(name, x)

@config_enumerate
def continuous_model(args, data):
    if False:
        i = 10
        return i + 15
    (rate_s, prob_i, rho) = global_model(args.population)
    S_aux = pyro.sample('S_aux', dist.Uniform(-0.5, args.population + 0.5).mask(False).expand(data.shape).to_event(1))
    I_aux = pyro.sample('I_aux', dist.Uniform(-0.5, args.population + 0.5).mask(False).expand(data.shape).to_event(1))
    S_curr = torch.tensor(args.population - 1.0)
    I_curr = torch.tensor(1.0)
    for (t, datum) in poutine.markov(enumerate(data)):
        (S_prev, I_prev) = (S_curr, I_curr)
        S_curr = quantize('S_{}'.format(t), S_aux[..., t], min=0, max=args.population)
        I_curr = quantize('I_{}'.format(t), I_aux[..., t], min=0, max=args.population)
        S2I = S_prev - S_curr
        I2R = I_prev - I_curr + S2I
        pyro.sample('S2I_{}'.format(t), dist.ExtendedBinomial(S_prev, -(rate_s * I_prev).expm1()), obs=S2I)
        pyro.sample('I2R_{}'.format(t), dist.ExtendedBinomial(I_prev, prob_i), obs=I2R)
        pyro.sample('obs_{}'.format(t), dist.ExtendedBinomial(S2I, rho), obs=datum)

def heuristic_init(args, data):
    if False:
        return 10
    'Heuristically initialize to a feasible point.'
    S0 = args.population - 1
    S2I = data * min(2.0, (S0 / data.sum()).sqrt())
    S_aux = (S0 - S2I.cumsum(-1)).clamp(min=0.5)
    S2I[0] += 1
    recovery = torch.arange(30.0).div(args.recovery_time).neg().exp()
    I_aux = convolve(S2I, recovery)[:len(data)].clamp(min=0.5)
    return {'R0': torch.tensor(2.0), 'rho': torch.tensor(0.5), 'S_aux': S_aux, 'I_aux': I_aux}

def infer_hmc_cont(model, args, data):
    if False:
        for i in range(10):
            print('nop')
    init_values = heuristic_init(args, data)
    return _infer_hmc(args, data, model, init_values=init_values)

def quantize_enumerate(x_real, min, max):
    if False:
        for i in range(10):
            print('nop')
    '\n    Randomly quantize in a way that preserves probability mass.\n    We use a piecewise polynomial spline of order 3.\n    '
    assert min < max
    lb = x_real.detach().floor()
    s = x_real - lb
    ss = s * s
    t = 1 - s
    tt = t * t
    probs = torch.stack([t * tt, 4 + ss * (3 * s - 6), 4 + tt * (3 * t - 6), s * ss], dim=-1) * (1 / 6)
    logits = safe_log(probs)
    q = torch.arange(-1.0, 3.0)
    x = lb.unsqueeze(-1) + q
    x = torch.max(x, 2 * min - 1 - x)
    x = torch.min(x, 2 * max + 1 - x)
    return (x, logits)

def vectorized_model(args, data):
    if False:
        i = 10
        return i + 15
    (rate_s, prob_i, rho) = global_model(args.population)
    S_aux = pyro.sample('S_aux', dist.Uniform(-0.5, args.population + 0.5).mask(False).expand(data.shape).to_event(1))
    I_aux = pyro.sample('I_aux', dist.Uniform(-0.5, args.population + 0.5).mask(False).expand(data.shape).to_event(1))
    (S_curr, S_logp) = quantize_enumerate(S_aux, min=0, max=args.population)
    (I_curr, I_logp) = quantize_enumerate(I_aux, min=0, max=args.population)
    S_prev = torch.nn.functional.pad(S_curr[:-1], (0, 0, 1, 0), value=args.population - 1)
    I_prev = torch.nn.functional.pad(I_curr[:-1], (0, 0, 1, 0), value=1)
    T = len(data)
    Q = 4
    S_prev = S_prev.reshape(T, Q, 1, 1, 1)
    I_prev = I_prev.reshape(T, 1, Q, 1, 1)
    S_curr = S_curr.reshape(T, 1, 1, Q, 1)
    S_logp = S_logp.reshape(T, 1, 1, Q, 1)
    I_curr = I_curr.reshape(T, 1, 1, 1, Q)
    I_logp = I_logp.reshape(T, 1, 1, 1, Q)
    data = data.reshape(T, 1, 1, 1, 1)
    S2I = S_prev - S_curr
    I2R = I_prev - I_curr + S2I
    S2I_logp = dist.ExtendedBinomial(S_prev, -(rate_s * I_prev).expm1()).log_prob(S2I)
    I2R_logp = dist.ExtendedBinomial(I_prev, prob_i).log_prob(I2R)
    obs_logp = dist.ExtendedBinomial(S2I, rho).log_prob(data)
    logp = S_logp + (I_logp + obs_logp) + S2I_logp + I2R_logp
    logp = logp.reshape(-1, Q * Q, Q * Q)
    logp = pyro.distributions.hmm._sequential_logmatmulexp(logp)
    logp = logp.reshape(-1).logsumexp(0)
    logp = logp - math.log(4)
    warn_if_nan(logp)
    pyro.factor('obs', logp)

def evaluate(args, samples):
    if False:
        for i in range(10):
            print('nop')
    names = {'basic_reproduction_number': 'R0', 'response_rate': 'rho'}
    for (name, key) in names.items():
        mean = samples[key].mean().item()
        std = samples[key].std().item()
        logging.info('{}: truth = {:0.3g}, estimate = {:0.3g} ± {:0.3g}'.format(key, getattr(args, name), mean, std))
    if args.plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        (fig, axes) = plt.subplots(2, 1, figsize=(5, 5))
        axes[0].set_title('Posterior parameter estimates')
        for (ax, (name, key)) in zip(axes, names.items()):
            truth = getattr(args, name)
            sns.distplot(samples[key], ax=ax, label='posterior')
            ax.axvline(truth, color='k', label='truth')
            ax.set_xlabel(key + ' = ' + name.replace('_', ' '))
            ax.set_yticks(())
            ax.legend(loc='best')
        plt.tight_layout()

@torch.no_grad()
def predict(args, data, samples, truth=None):
    if False:
        i = 10
        return i + 15
    logging.info('Forecasting {} steps ahead...'.format(args.forecast))
    particle_plate = pyro.plate('particles', args.num_samples, dim=-1)
    model = poutine.condition(continuous_model, samples)
    model = particle_plate(model)
    model = infer_discrete(model, first_available_dim=-2)
    with poutine.trace() as tr:
        model(args, data)
    samples = OrderedDict(((name, site['value']) for (name, site) in tr.trace.nodes.items() if site['type'] == 'sample'))
    extended_data = list(data) + [None] * args.forecast
    model = poutine.condition(discrete_model, samples)
    model = particle_plate(model)
    with poutine.trace() as tr:
        model(args, extended_data)
    samples = OrderedDict(((name, site['value']) for (name, site) in tr.trace.nodes.items() if site['type'] == 'sample'))
    for key in ('S', 'I', 'S2I', 'I2R'):
        pattern = key + '_[0-9]+'
        series = [value for (name, value) in samples.items() if re.match(pattern, name)]
        assert len(series) == args.duration + args.forecast
        series[0] = series[0].expand(series[1].shape)
        samples[key] = torch.stack(series, dim=-1)
    S2I = samples['S2I']
    median = S2I.median(dim=0).values
    logging.info('Median prediction of new infections (starting on day 0):\n{}'.format(' '.join(map(str, map(int, median)))))
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        time = torch.arange(args.duration + args.forecast)
        p05 = S2I.kthvalue(int(round(0.5 + 0.05 * args.num_samples)), dim=0).values
        p95 = S2I.kthvalue(int(round(0.5 + 0.95 * args.num_samples)), dim=0).values
        plt.fill_between(time, p05, p95, color='red', alpha=0.3, label='90% CI')
        plt.plot(time, median, 'r-', label='median')
        plt.plot(time[:args.duration], data, 'k.', label='observed')
        if truth is not None:
            plt.plot(time, truth, 'k--', label='truth')
        plt.axvline(args.duration - 0.5, color='gray', lw=1)
        plt.xlim(0, len(time) - 1)
        plt.ylim(0, None)
        plt.xlabel('day after first infection')
        plt.ylabel('new infections per day')
        plt.title('New infections in population of {}'.format(args.population))
        plt.legend(loc='upper left')
        plt.tight_layout()
    return samples

def main(args):
    if False:
        while True:
            i = 10
    pyro.set_rng_seed(args.rng_seed)
    dataset = generate_data(args)
    obs = dataset['obs'][:args.duration]
    if args.enum:
        samples = infer_hmc_enum(args, obs)
    elif args.sequential:
        samples = infer_hmc_cont(continuous_model, args, obs)
    else:
        samples = infer_hmc_cont(vectorized_model, args, obs)
    evaluate(args, samples)
    if args.forecast:
        samples = predict(args, obs, samples, truth=dataset['S2I'])
    return samples
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='SIR epidemiology modeling using HMC')
    parser.add_argument('-p', '--population', default=10, type=int)
    parser.add_argument('-m', '--min-observations', default=3, type=int)
    parser.add_argument('-d', '--duration', default=10, type=int)
    parser.add_argument('-f', '--forecast', default=0, type=int)
    parser.add_argument('-R0', '--basic-reproduction-number', default=1.5, type=float)
    parser.add_argument('-tau', '--recovery-time', default=7.0, type=float)
    parser.add_argument('-rho', '--response-rate', default=0.5, type=float)
    parser.add_argument('-e', '--enum', action='store_true', help='use the full enumeration model')
    parser.add_argument('-s', '--sequential', action='store_true', help='use the sequential continuous model')
    parser.add_argument('-n', '--num-samples', default=200, type=int)
    parser.add_argument('-w', '--warmup-steps', default=100, type=int)
    parser.add_argument('-t', '--max-tree-depth', default=5, type=int)
    parser.add_argument('-r', '--rng-seed', default=0, type=int)
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.double:
        torch.set_default_dtype(torch.float64)
    if args.cuda:
        torch.set_default_device('cuda')
    main(args)
    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()