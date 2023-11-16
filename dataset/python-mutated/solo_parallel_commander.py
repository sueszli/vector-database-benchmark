from typing import Optional
import time
import copy
from ding.policy import create_policy
from ding.utils import LimitedSpaceContainer, get_task_uid, build_logger, COMMANDER_REGISTRY
from .base_parallel_commander import BaseCommander

@COMMANDER_REGISTRY.register('solo')
class SoloCommander(BaseCommander):
    """
    Overview:
        Parallel commander for solo games.
    Interface:
        __init__, get_collector_task, get_learner_task, finish_collector_task, finish_learner_task,
        notify_fail_collector_task, notify_fail_learner_task, update_learner_info
    """
    config = dict(collector_task_space=1, learner_task_space=1, eval_interval=60)

    def __init__(self, cfg: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Init the solo commander according to config.\n        Arguments:\n            - cfg (:obj:`dict`): Dict type config file.\n        '
        self._cfg = cfg
        self._exp_name = cfg.exp_name
        commander_cfg = self._cfg.policy.other.commander
        self._commander_cfg = commander_cfg
        self._collector_env_cfg = copy.deepcopy(self._cfg.env)
        self._collector_env_cfg.pop('collector_episode_num')
        self._collector_env_cfg.pop('evaluator_episode_num')
        self._collector_env_cfg.manager.episode_num = self._cfg.env.collector_episode_num
        self._evaluator_env_cfg = copy.deepcopy(self._cfg.env)
        self._evaluator_env_cfg.pop('collector_episode_num')
        self._evaluator_env_cfg.pop('evaluator_episode_num')
        self._evaluator_env_cfg.manager.episode_num = self._cfg.env.evaluator_episode_num
        self._collector_task_space = LimitedSpaceContainer(0, commander_cfg.collector_task_space)
        self._learner_task_space = LimitedSpaceContainer(0, commander_cfg.learner_task_space)
        self._learner_info = [{'learner_step': 0}]
        self._collector_info = []
        self._total_collector_env_step = 0
        self._evaluator_info = []
        self._current_buffer_id = None
        self._current_policy_id = None
        self._last_eval_time = 0
        policy_cfg = copy.deepcopy(self._cfg.policy)
        self._policy = create_policy(policy_cfg, enable_field=['command']).command_mode
        (self._logger, self._tb_logger) = build_logger('./{}/log/commander'.format(self._exp_name), 'commander', need_tb=True)
        (self._collector_logger, _) = build_logger('./{}/log/commander'.format(self._exp_name), 'commander_collector', need_tb=False)
        (self._evaluator_logger, _) = build_logger('./{}/log/commander'.format(self._exp_name), 'commander_evaluator', need_tb=False)
        self._sub_logger = {'collector': self._collector_logger, 'evaluator': self._evaluator_logger}
        self._end_flag = False

    def get_collector_task(self) -> Optional[dict]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Return the new collector task when there is residual task space; Otherwise return None.\n        Return:\n            - task (:obj:`Optional[dict]`): New collector task.\n        '
        if self._end_flag:
            return None
        if self._collector_task_space.acquire_space():
            if self._current_buffer_id is None or self._current_policy_id is None:
                self._collector_task_space.release_space()
                return None
            cur_time = time.time()
            if cur_time - self._last_eval_time > self._commander_cfg.eval_interval:
                eval_flag = True
                self._last_eval_time = time.time()
            else:
                eval_flag = False
            collector_cfg = copy.deepcopy(self._cfg.policy.collect.collector)
            info = self._learner_info[-1]
            info['envstep'] = self._total_collector_env_step
            collector_cfg.collect_setting = self._policy.get_setting_collect(info)
            collector_cfg.policy_update_path = self._current_policy_id
            collector_cfg.eval_flag = eval_flag
            collector_cfg.policy = copy.deepcopy(self._cfg.policy)
            collector_cfg.exp_name = self._exp_name
            if eval_flag:
                collector_cfg.env = self._evaluator_env_cfg
            else:
                collector_cfg.env = self._collector_env_cfg
            return {'task_id': 'collector_task_{}'.format(get_task_uid()), 'buffer_id': self._current_buffer_id, 'collector_cfg': collector_cfg}
        else:
            return None

    def get_learner_task(self) -> Optional[dict]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Return the new learner task when there is residual task space; Otherwise return None.\n        Return:\n            - task (:obj:`Optional[dict]`): New learner task.\n        '
        if self._end_flag:
            return None
        if self._learner_task_space.acquire_space():
            learner_cfg = copy.deepcopy(self._cfg.policy.learn.learner)
            learner_cfg.exp_name = self._exp_name
            return {'task_id': 'learner_task_{}'.format(get_task_uid()), 'policy_id': self._init_policy_id(), 'buffer_id': self._init_buffer_id(), 'learner_cfg': learner_cfg, 'replay_buffer_cfg': copy.deepcopy(self._cfg.policy.other.replay_buffer), 'policy': copy.deepcopy(self._cfg.policy)}
        else:
            return None

    def finish_collector_task(self, task_id: str, finished_task: dict) -> bool:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Get collector's finish_task_info and release collector_task_space.\n            If collector's task is evaluation, judge the convergence and return it.\n        Arguments:\n            - task_id (:obj:`str`): the collector task_id\n            - finished_task (:obj:`dict`): the finished task\n        Returns:\n            - convergence (:obj:`bool`): Whether the stop val is reached and the algorithm is converged. \\\n                If True, the pipeline can be finished.\n        "
        self._collector_task_space.release_space()
        evaluator_or_collector = 'evaluator' if finished_task['eval_flag'] else 'collector'
        train_iter = finished_task['train_iter']
        info = {'train_iter': train_iter, 'episode_count': finished_task['real_episode_count'], 'step_count': finished_task['step_count'], 'avg_step_per_episode': finished_task['avg_time_per_episode'], 'avg_time_per_step': finished_task['avg_time_per_step'], 'avg_time_per_episode': finished_task['avg_step_per_episode'], 'reward_mean': finished_task['reward_mean'], 'reward_std': finished_task['reward_std']}
        self._sub_logger[evaluator_or_collector].info('[{}] Task ends:\n{}'.format(evaluator_or_collector.upper(), '\n'.join(['{}: {}'.format(k, v) for (k, v) in info.items()])))
        for (k, v) in info.items():
            if k in ['train_iter']:
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(evaluator_or_collector) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(evaluator_or_collector) + k, v, self._total_collector_env_step)
        if finished_task['eval_flag']:
            self._evaluator_info.append(finished_task)
            eval_stop_value = self._cfg.env.stop_value
            if eval_stop_value is not None and finished_task['reward_mean'] >= eval_stop_value:
                self._logger.info('[DI-engine parallel pipeline] current episode_return: {} is greater than the stop_value: {}'.format(finished_task['reward_mean'], eval_stop_value) + ', so the total training program is over.')
                self._end_flag = True
                return True
        else:
            self._collector_info.append(finished_task)
            self._total_collector_env_step += finished_task['step_count']
        return False

    def finish_learner_task(self, task_id: str, finished_task: dict) -> str:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Get learner's finish_task_info, release learner_task_space, reset corresponding variables.\n        Arguments:\n            - task_id (:obj:`str`): Learner task_id\n            - finished_task (:obj:`dict`): Learner's finish_learn_info.\n        Returns:\n            - buffer_id (:obj:`str`): Buffer id of the finished learner.\n        "
        self._learner_task_space.release_space()
        buffer_id = finished_task['buffer_id']
        self._current_buffer_id = None
        self._current_policy_id = None
        self._learner_info = [{'learner_step': 0}]
        self._evaluator_info = []
        self._last_eval_time = 0
        return buffer_id

    def notify_fail_collector_task(self, task: dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Release task space when collector task fails.\n        '
        self._collector_task_space.release_space()

    def notify_fail_learner_task(self, task: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Release task space when learner task fails.\n        '
        self._learner_task_space.release_space()

    def update_learner_info(self, task_id: str, info: dict) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Append the info to learner_info:\n        Arguments:\n            - task_id (:obj:`str`): Learner task_id\n            - info (:obj:`dict`): Dict type learner info.\n        '
        self._learner_info.append(info)

    def _init_policy_id(self) -> str:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Init the policy id and return it.\n        Returns:\n            - policy_id (:obj:`str`): New initialized policy id.\n        '
        policy_id = 'policy_{}'.format(get_task_uid())
        self._current_policy_id = policy_id
        return policy_id

    def _init_buffer_id(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Init the buffer id and return it.\n        Returns:\n            - buffer_id (:obj:`str`): New initialized buffer id.\n        '
        buffer_id = 'buffer_{}'.format(get_task_uid())
        self._current_buffer_id = buffer_id
        return buffer_id

    def increase_collector_task_space(self):
        if False:
            while True:
                i = 10
        '"\n        Overview:\n        Increase task space when a new collector has added dynamically.\n        '
        self._collector_task_space.increase_space()

    def decrease_collector_task_space(self):
        if False:
            i = 10
            return i + 15
        '"\n        Overview:\n        Decrease task space when a new collector has removed dynamically.\n        '
        self._collector_task_space.decrease_space()