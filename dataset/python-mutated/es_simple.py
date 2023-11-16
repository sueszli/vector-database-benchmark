import numpy as np
import matplotlib.pyplot as plt

def evolution_strategy(f, population_size, sigma, lr, initial_params, num_iters):
    if False:
        return 10
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)
    params = initial_params
    for t in range(num_iters):
        N = np.random.randn(population_size, num_params)
        R = np.zeros(population_size)
        for j in range(population_size):
            params_try = params + sigma * N[j]
            R[j] = f(params_try)
        m = R.mean()
        A = (R - m) / R.std()
        reward_per_iteration[t] = m
        params = params + lr / (population_size * sigma) * np.dot(N.T, A)
    return (params, reward_per_iteration)

def reward_function(params):
    if False:
        i = 10
        return i + 15
    x0 = params[0]
    x1 = params[1]
    x2 = params[2]
    return -(x0 ** 2 + 0.1 * (x1 - 1) ** 2 + 0.5 * (x2 + 2) ** 2)
if __name__ == '__main__':
    (best_params, rewards) = evolution_strategy(f=reward_function, population_size=50, sigma=0.1, lr=0.001, initial_params=np.random.randn(3), num_iters=500)
    plt.plot(rewards)
    plt.show()
    print('Final params:', best_params)