import gymnasium as gym

def create_cartpole_deterministic(config):
    if False:
        while True:
            i = 10
    env = gym.make('CartPole-v1')
    env.reset(seed=config.get('seed', 0))
    return env

def create_pendulum_deterministic(config):
    if False:
        return 10
    env = gym.make('Pendulum-v1')
    env.reset(seed=config.get('seed', 0))
    return env