import os
import gym
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import DDPGPolicy
from ding.model import ContinuousQAC
from ding.utils import set_pkg_seed
from dizoo.classic_control.pendulum.envs import PendulumEnv
from dizoo.classic_control.pendulum.config.pendulum_td3_config import pendulum_td3_config

def main(cfg, seed=0):
    if False:
        i = 10
        return i + 15
    cfg = compile_config(cfg, BaseEnvManager, DDPGPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer, save_cfg=True)
    (collector_env_num, evaluator_env_num) = (cfg.env.collector_env_num, cfg.env.evaluator_env_num)
    collector_env = BaseEnvManager(env_fn=[lambda : PendulumEnv(cfg.env) for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[lambda : PendulumEnv(cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    model = ContinuousQAC(**cfg.policy.model)
    policy = DDPGPolicy(cfg.policy, model=model)
    lr_scheduler = LambdaLR(policy.learn_mode.get_attribute('optimizer_actor'), lr_lambda=lambda iters: min(1.0, 0.5 + 0.5 * iters / 1000))
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
            lr_scheduler.step()
        tb_logger.add_scalar('other_iter/scheduled_lr', lr_scheduler.get_last_lr()[0], learner.train_iter)
if __name__ == '__main__':
    main(pendulum_td3_config, seed=0)