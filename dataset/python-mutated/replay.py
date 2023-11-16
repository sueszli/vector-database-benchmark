from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import str
from builtins import object
import numpy as np
import pickle
import multiprocessing

class ReplayBuffer(object):
    """
    Stores frames sampled from the environment, with the ability to sample a batch
    for training.
    """

    def __init__(self, max_size, obs_dim, action_dim, roundrobin=True):
        if False:
            print('Hello World!')
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.roundrobin = roundrobin
        self.obs_buffer = np.zeros([max_size, obs_dim])
        self.next_obs_buffer = np.zeros([max_size, obs_dim])
        self.action_buffer = np.zeros([max_size, action_dim])
        self.reward_buffer = np.zeros([max_size])
        self.done_buffer = np.zeros([max_size])
        self.count = 0

    def random_batch(self, batch_size):
        if False:
            while True:
                i = 10
        indices = np.random.randint(0, min(self.count, self.max_size), batch_size)
        return (self.obs_buffer[indices], self.next_obs_buffer[indices], self.action_buffer[indices], self.reward_buffer[indices], self.done_buffer[indices], self.count)

    def add_replay(self, obs, next_obs, action, reward, done):
        if False:
            while True:
                i = 10
        if self.count >= self.max_size:
            if self.roundrobin:
                index = self.count % self.max_size
            else:
                index = np.random.randint(0, self.max_size)
        else:
            index = self.count
        self.obs_buffer[index] = obs
        self.next_obs_buffer[index] = next_obs
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.done_buffer[index] = done
        self.count += 1

    def save(self, path, name):
        if False:
            return 10

        def _save(datas, fnames):
            if False:
                i = 10
                return i + 15
            print('saving replay buffer...')
            for (data, fname) in zip(datas, fnames):
                with open('%s.npz' % fname, 'w') as f:
                    pickle.dump(data, f)
            with open('%s/%s.count' % (path, name), 'w') as f:
                f.write(str(self.count))
            print('...done saving.')
        datas = [self.obs_buffer, self.next_obs_buffer, self.action_buffer, self.reward_buffer, self.done_buffer]
        fnames = ['%s/%s.obs_buffer' % (path, name), '%s/%s.next_obs_buffer' % (path, name), '%s/%s.action_buffer' % (path, name), '%s/%s.reward_buffer' % (path, name), '%s/%s.done_buffer' % (path, name)]
        proc = multiprocessing.Process(target=_save, args=(datas, fnames))
        proc.start()

    def load(self, path, name):
        if False:
            print('Hello World!')
        print('Loading %s replay buffer (may take a while...)' % name)
        with open('%s/%s.obs_buffer.npz' % (path, name)) as f:
            self.obs_buffer = pickle.load(f)
        with open('%s/%s.next_obs_buffer.npz' % (path, name)) as f:
            self.next_obs_buffer = pickle.load(f)
        with open('%s/%s.action_buffer.npz' % (path, name)) as f:
            self.action_buffer = pickle.load(f)
        with open('%s/%s.reward_buffer.npz' % (path, name)) as f:
            self.reward_buffer = pickle.load(f)
        with open('%s/%s.done_buffer.npz' % (path, name)) as f:
            self.done_buffer = pickle.load(f)
        with open('%s/%s.count' % (path, name), 'r') as f:
            self.count = int(f.read())