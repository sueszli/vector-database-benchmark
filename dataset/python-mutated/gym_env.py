from ding.envs import BaseEnv, DingEnvWrapper

def env(cfg, seed_api=True, caller='collector', **kwargs) -> BaseEnv:
    if False:
        return 10
    import gym
    return DingEnvWrapper(gym.make(cfg.env_id, **kwargs), cfg=cfg, seed_api=seed_api, caller=caller)