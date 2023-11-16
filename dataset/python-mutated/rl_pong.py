from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import time
import gym
import numpy as np
import ray
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
os.environ['LANG'] = 'C.UTF-8'
H = 200
learning_rate = 0.0001
gamma = 0.99
decay_rate = 0.99
D = 80 * 80

def sigmoid(x):
    if False:
        for i in range(10):
            print('nop')
    return 1.0 / (1.0 + np.exp(-x))

def preprocess(img):
    if False:
        return 10
    'Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.'
    img = img[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    return img.astype(np.float).ravel()

def discount_rewards(r):
    if False:
        print('Hello World!')
    'take 1D float array of rewards and compute discounted reward'
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x, model):
    if False:
        while True:
            i = 10
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return (p, h)

def policy_backward(eph, epx, epdlogp, model):
    if False:
        return 10
    'backward pass. (eph is array of intermediate hidden states)'
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

@ray.remote
class PongEnv(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        os.environ['MKL_NUM_THREADS'] = '1'
        self.env = gym.make('Pong-v0')

    def compute_gradient(self, model):
        if False:
            i = 10
            return i + 15
        observation = self.env.reset()
        prev_x = None
        (xs, hs, dlogps, drs) = ([], [], [], [])
        reward_sum = 0
        done = False
        while not done:
            cur_x = preprocess(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(D)
            prev_x = cur_x
            (aprob, h) = policy_forward(x, model)
            action = 2 if np.random.uniform() < aprob else 3
            xs.append(x)
            hs.append(h)
            y = 1 if action == 2 else 0
            dlogps.append(y - aprob)
            (observation, reward, done, info) = self.env.step(action)
            reward_sum += reward
            drs.append(reward)
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        (xs, hs, dlogps, drs) = ([], [], [], [])
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr
        return (policy_backward(eph, epx, epdlogp, model), reward_sum)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an RL agent')
    parser.add_argument('--cluster_mode', type=str, default='local', help='The mode for the Spark cluster. local, yarn or spark-submit.')
    parser.add_argument('--batch_size', default=10, type=int, help='The number of roll-outs to do per batch.')
    parser.add_argument('--iterations', default=-1, type=int, help='The number of model updates to perform. By default, training will not terminate.')
    parser.add_argument('--slave_num', type=int, default=2, help='The number of slave nodes')
    parser.add_argument('--executor_cores', type=int, default=8, help="The number of driver's cpu cores you want to use.You can change it depending on your own cluster setting.")
    parser.add_argument('--executor_memory', type=str, default='10g', help="The size of slave(executor)'s memory you want to use.You can change it depending on your own cluster setting.")
    parser.add_argument('--driver_memory', type=str, default='2g', help="The size of driver's memory you want to use.You can change it depending on your own cluster setting.")
    parser.add_argument('--driver_cores', type=int, default=8, help="The number of driver's cpu cores you want to use.You can change it depending on your own cluster setting.")
    parser.add_argument('--extra_executor_memory_for_ray', type=str, default='20g', help='The extra executor memory to store some data.You can change it depending on your own cluster setting.')
    parser.add_argument('--object_store_memory', type=str, default='4g', help='The memory to store data on local.You can change it depending on your own cluster setting.')
    args = parser.parse_args()
    cluster_mode = args.cluster_mode
    if cluster_mode.startswith('yarn'):
        sc = init_orca_context(cluster_mode=cluster_mode, cores=args.executor_cores, memory=args.executor_memory, init_ray_on_spark=True, num_executors=args.slave_num, driver_memory=args.driver_memory, driver_cores=args.driver_cores, extra_executor_memory_for_ray=args.extra_executor_memory_for_ray, object_store_memory=args.object_store_memory)
        ray_ctx = OrcaContext.get_ray_context()
    elif cluster_mode == 'local':
        sc = init_orca_context(cores=args.driver_cores)
        ray_ctx = OrcaContext.get_ray_context()
    elif cluster_mode == 'spark-submit':
        sc = init_orca_context(cluster_mode=cluster_mode)
        ray_ctx = OrcaContext.get_ray_context()
    else:
        print("init_orca_context failed. cluster_mode should be one of 'local','yarn' and 'spark-submit' but got " + cluster_mode)
    batch_size = args.batch_size
    running_reward = None
    batch_num = 1
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)
    grad_buffer = {k: np.zeros_like(v) for (k, v) in model.items()}
    rmsprop_cache = {k: np.zeros_like(v) for (k, v) in model.items()}
    actors = [PongEnv.remote() for _ in range(batch_size)]
    iteration = 0
    while iteration != args.iterations:
        iteration += 1
        model_id = ray.put(model)
        actions = []
        start_time = time.time()
        for i in range(batch_size):
            action_id = actors[i].compute_gradient.remote(model_id)
            actions.append(action_id)
        for i in range(batch_size):
            (action_id, actions) = ray.wait(actions)
            (grad, reward_sum) = ray.get(action_id[0])
            for k in model:
                grad_buffer[k] += grad[k]
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        end_time = time.time()
        print('Batch {} computed {} rollouts in {} seconds, running mean is {}'.format(batch_num, batch_size, end_time - start_time, running_reward))
        for (k, v) in model.items():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-05)
            grad_buffer[k] = np.zeros_like(v)
        batch_num += 1
    stop_orca_context()