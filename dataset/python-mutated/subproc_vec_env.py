import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
    if False:
        i = 10
        return i + 15
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        (cmd, data) = remote.recv()
        if cmd == 'step':
            (ob, reward, done, info) = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'get_id':
            remote.send(env.spec.id)
        else:
            raise NotImplementedError

class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.x = x

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        if False:
            while True:
                i = 10
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv:

    def __init__(self, env_fns):
        if False:
            print('Hello World!')
        '\n        envs: list of gym environments to run in subprocesses\n        '
        self.closed = False
        nenvs = len(env_fns)
        (self.remotes, self.work_remotes) = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))) for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        (self.action_space, self.observation_space) = self.remotes[0].recv()
        self.remotes[0].send(('get_id', None))
        self.env_id = self.remotes[0].recv()

    def step(self, actions):
        if False:
            while True:
                i = 10
        for (remote, action) in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        (obs, rews, dones, infos) = zip(*results)
        return (np.stack(obs), np.stack(rews), np.stack(dones), infos)

    def reset(self):
        if False:
            while True:
                i = 10
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        if False:
            while True:
                i = 10
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if False:
            print('Hello World!')
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        if False:
            while True:
                i = 10
        return len(self.remotes)