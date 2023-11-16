import os
import gym
from tensorboardX import SummaryWriter
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import DDPGPolicy
from ding.model import ContinuousQAC
from ding.utils import set_pkg_seed
from dizoo.classic_control.pendulum.envs import PendulumEnv
from dizoo.classic_control.pendulum.config.pendulum_ddpg_config import pendulum_ddpg_config

def main(cfg, seed=0):
    if False:
        while True:
            i = 10
    cfg = compile_config(cfg, BaseEnvManager, DDPGPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer, save_cfg=True)
    (collector_env_num, evaluator_env_num) = (cfg.env.collector_env_num, cfg.env.evaluator_env_num)
    collector_env = BaseEnvManager(env_fn=[lambda : DingEnvWrapper(env=gym.make('Pendulum-v1'), cfg=cfg.env) for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[lambda : DingEnvWrapper(env=gym.make('Pendulum-v1'), cfg=cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    model = ContinuousQAC(**cfg.policy.model)
    policy = DDPGPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    while True:
        if evaluator.should_eval(learner.train_iter):
            (stop, reward) = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
if __name__ == '__main__':
    main(pendulum_ddpg_config, seed=0)