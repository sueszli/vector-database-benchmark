from typing import Union, Optional, List, Any, Tuple
import os
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed
from .utils import random_collect

def serial_pipeline_ngu(input_cfg: Union[str, Tuple[dict, dict]], seed: int=0, env_setting: Optional[List[Any]]=None, model: Optional[torch.nn.Module]=None, max_train_iter: Optional[int]=int(10000000000.0), max_env_step: Optional[int]=int(10000000000.0)) -> 'Policy':
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Serial pipeline entry for NGU. The corresponding paper is\n        `never give up: learning directed exploration strategies`.\n    Arguments:\n        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type.             ``str`` type means config file path.             ``Tuple[dict, dict]`` type means [user_config, create_cfg].\n        - seed (:obj:`int`): Random seed.\n        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements:             ``BaseEnv`` subclass, collector env config, and evaluator env config.\n        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.\n        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.\n        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.\n    Returns:\n        - policy (:obj:`Policy`): Converged policy.\n    '
    if isinstance(input_cfg, str):
        (cfg, create_cfg) = read_config(input_cfg)
    else:
        (cfg, create_cfg) = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    if env_setting is None:
        (env_fn, collector_env_cfg, evaluator_env_cfg) = get_vec_env_setting(cfg.env)
    else:
        (env_fn, collector_env_cfg, evaluator_env_cfg) = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(cfg.policy.collect.collector, env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode)
    rnd_reward_model = create_reward_model(cfg.rnd_reward_model, policy.collect_mode.get_attribute('device'), tb_logger)
    episodic_reward_model = create_reward_model(cfg.episodic_reward_model, policy.collect_mode.get_attribute('device'), tb_logger)
    learner.call_hook('before_run')
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)
    estimate_cnt = 0
    iter_ = 0
    while True:
        'some hyper-parameters used in NGU'
        iter_ += 1
        if evaluator.should_eval(learner.train_iter):
            (stop, reward) = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=None)
        rnd_reward_model.collect_data(new_data)
        episodic_reward_model.collect_data(new_data)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        rnd_reward_model.train()
        if (iter_ + 1) % cfg.rnd_reward_model.clear_buffer_per_iters == 0:
            rnd_reward_model.clear_data()
        episodic_reward_model.train()
        if (iter_ + 1) % cfg.episodic_reward_model.clear_buffer_per_iters == 0:
            episodic_reward_model.clear_data()
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                logging.warning("Replay buffer's data can only train for {} steps. ".format(i) + 'You can modify data collect config, e.g. increasing n_sample, n_episode.')
                break
            rnd_reward = rnd_reward_model.estimate(train_data)
            episodic_reward = episodic_reward_model.estimate(train_data)
            (train_data_augmented, estimate_cnt) = episodic_reward_model.fusion_reward(train_data, rnd_reward, episodic_reward, nstep=cfg.policy.nstep, collector_env_num=cfg.policy.collect.env_num, tb_logger=tb_logger, estimate_cnt=estimate_cnt)
            learner.train(train_data_augmented, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break
    learner.call_hook('after_run')
    return policy