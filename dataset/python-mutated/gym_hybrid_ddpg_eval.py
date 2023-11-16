import os
import gym
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict
from functools import partial
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.envs import get_vec_env_setting
from ding.policy import DDPGPolicy
from ding.model import ContinuousQAC
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.gym_hybrid.config.gym_hybrid_ddpg_config import gym_hybrid_ddpg_config, gym_hybrid_ddpg_create_config

def main(main_cfg, create_cfg, seed=0):
    if False:
        return 10
    main_cfg.policy.load_path = './ckpt_best.pth.tar'
    main_cfg.env.replay_path = './'
    main_cfg.env.evaluator_env_num = 1
    cfg = compile_config(main_cfg, seed=seed, auto=True, create_cfg=create_cfg, save_cfg=True)
    (env_fn, collector_env_cfg, evaluator_env_cfg) = get_vec_env_setting(cfg.env)
    evaluator_env = BaseEnvManager([partial(env_fn, cfg=c) for c in evaluator_env_cfg], cfg.env.manager)
    evaluator_env.enable_save_replay(cfg.env.replay_path)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    model = ContinuousQAC(**cfg.policy.model)
    policy = DDPGPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator.eval()
if __name__ == '__main__':
    main(gym_hybrid_ddpg_config, gym_hybrid_ddpg_create_config, seed=0)