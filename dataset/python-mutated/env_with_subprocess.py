import atexit
import gymnasium as gym
from gymnasium.spaces import Discrete
import os
import subprocess

class EnvWithSubprocess(gym.Env):
    """An env that spawns a subprocess."""
    UNIQUE_CMD = 'sleep 20'

    def __init__(self, config):
        if False:
            return 10
        self.UNIQUE_FILE_0 = config['tmp_file1']
        self.UNIQUE_FILE_1 = config['tmp_file2']
        self.UNIQUE_FILE_2 = config['tmp_file3']
        self.UNIQUE_FILE_3 = config['tmp_file4']
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)
        self.subproc = subprocess.Popen(self.UNIQUE_CMD.split(' '), shell=False)
        self.config = config
        atexit.register(lambda : self.subproc.kill())
        if config.worker_index == 0:
            atexit.register(lambda : os.unlink(self.UNIQUE_FILE_0))
        else:
            atexit.register(lambda : os.unlink(self.UNIQUE_FILE_1))

    def close(self):
        if False:
            while True:
                i = 10
        if self.config.worker_index == 0:
            os.unlink(self.UNIQUE_FILE_2)
        else:
            os.unlink(self.UNIQUE_FILE_3)

    def reset(self, *, seed=None, options=None):
        if False:
            return 10
        return (0, {})

    def step(self, action):
        if False:
            print('Hello World!')
        return (0, 0, True, False, {})