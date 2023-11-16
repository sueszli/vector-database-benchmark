import argparse
import gym
from stable_baselines.deepq import DQN

def main(args):
    if False:
        i = 10
        return i + 15
    '\n    Train and save the DQN model, for the mountain car problem\n\n    :param args: (ArgumentParser) the input arguments\n    '
    env = gym.make('MountainCar-v0')
    model = DQN(policy='LnMlpPolicy', env=env, learning_rate=0.001, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.1, param_noise=True, policy_kwargs=dict(layers=[64]))
    model.learn(total_timesteps=args.max_timesteps)
    print('Saving model to mountaincar_model.zip')
    model.save('mountaincar_model')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN on MountainCar')
    parser.add_argument('--max-timesteps', default=100000, type=int, help='Maximum number of timesteps')
    args = parser.parse_args()
    main(args)