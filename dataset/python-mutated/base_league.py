from typing import Union, Dict
import uuid
import copy
import os
import os.path as osp
from abc import abstractmethod
from easydict import EasyDict
from tabulate import tabulate
from ding.league.player import ActivePlayer, HistoricalPlayer, create_player
from ding.league.shared_payoff import create_payoff
from ding.utils import import_module, read_file, save_file, LockContext, LockContextType, LEAGUE_REGISTRY, deep_merge_dicts
from .metric import LeagueMetricEnv

class BaseLeague:
    """
    Overview:
        League, proposed by Google Deepmind AlphaStar. Can manage multiple players in one league.
    Interface:
        get_job_info, judge_snapshot, update_active_player, finish_job, save_checkpoint

    .. note::
        In ``__init__`` method, league would also initialized players as well(in ``_init_players`` method).
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            while True:
                i = 10
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    config = dict(league_type='base', import_names=['ding.league.base_league'], player_category=['default'], use_pretrain=False, use_pretrain_init_historical=False, pretrain_checkpoint_path=dict(default='default_cate_pretrain.pth'), payoff=dict(type='battle', decay=0.99, min_win_rate_games=8), metric=dict(mu=0, sigma=25 / 3, beta=25 / 3 / 2, tau=0.0, draw_probability=0.02))

    def __init__(self, cfg: EasyDict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Initialization method.\n        Arguments:\n            - cfg (:obj:`EasyDict`): League config.\n        '
        self.cfg = deep_merge_dicts(self.default_config(), cfg)
        self.path_policy = cfg.path_policy
        if not osp.exists(self.path_policy):
            os.mkdir(self.path_policy)
        self.league_uid = str(uuid.uuid1())
        self.active_players = []
        self.historical_players = []
        self.player_path = './league'
        self.payoff = create_payoff(self.cfg.payoff)
        metric_cfg = self.cfg.metric
        self.metric_env = LeagueMetricEnv(metric_cfg.mu, metric_cfg.sigma, metric_cfg.tau, metric_cfg.draw_probability)
        self._active_players_lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._init_players()

    def _init_players(self) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Initialize players (active & historical) in the league.\n        '
        for cate in self.cfg.player_category:
            for (k, n) in self.cfg.active_players.items():
                for i in range(n):
                    name = '{}_{}_{}'.format(k, cate, i)
                    ckpt_path = osp.join(self.path_policy, '{}_ckpt.pth'.format(name))
                    player = create_player(self.cfg, k, self.cfg[k], cate, self.payoff, ckpt_path, name, 0, self.metric_env.create_rating())
                    if self.cfg.use_pretrain:
                        self.save_checkpoint(self.cfg.pretrain_checkpoint_path[cate], ckpt_path)
                    self.active_players.append(player)
                    self.payoff.add_player(player)
        if self.cfg.use_pretrain_init_historical:
            for cate in self.cfg.player_category:
                main_player_name = [k for k in self.cfg.keys() if 'main_player' in k]
                assert len(main_player_name) == 1, main_player_name
                main_player_name = main_player_name[0]
                name = '{}_{}_0_pretrain_historical'.format(main_player_name, cate)
                parent_name = '{}_{}_0'.format(main_player_name, cate)
                hp = HistoricalPlayer(self.cfg.get(main_player_name), cate, self.payoff, self.cfg.pretrain_checkpoint_path[cate], name, 0, self.metric_env.create_rating(), parent_id=parent_name)
                self.historical_players.append(hp)
                self.payoff.add_player(hp)
        self.active_players_ids = [p.player_id for p in self.active_players]
        self.active_players_ckpts = [p.checkpoint_path for p in self.active_players]
        assert len(self.active_players_ids) == len(set(self.active_players_ids))

    def get_job_info(self, player_id: str=None, eval_flag: bool=False) -> dict:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Get info dict of the job which is to be launched to an active player.\n        Arguments:\n            - player_id (:obj:`str`): The active player's id.\n            - eval_flag (:obj:`bool`): Whether this is an evaluation job.\n        Returns:\n            - job_info (:obj:`dict`): Job info.\n        ReturnsKeys:\n            - necessary: ``launch_player`` (the active player)\n        "
        if player_id is None:
            player_id = self.active_players_ids[0]
        with self._active_players_lock:
            idx = self.active_players_ids.index(player_id)
            player = self.active_players[idx]
            job_info = self._get_job_info(player, eval_flag)
            assert 'launch_player' in job_info.keys() and job_info['launch_player'] == player.player_id
        return job_info

    @abstractmethod
    def _get_job_info(self, player: ActivePlayer, eval_flag: bool=False) -> dict:
        if False:
            return 10
        "\n        Overview:\n            Real `get_job` method. Called by ``_launch_job``.\n        Arguments:\n            - player (:obj:`ActivePlayer`): The active player to be launched a job.\n            - eval_flag (:obj:`bool`): Whether this is an evaluation job.\n        Returns:\n            - job_info (:obj:`dict`): Job info. Should include keys ['lauch_player'].\n        "
        raise NotImplementedError

    def judge_snapshot(self, player_id: str, force: bool=False) -> bool:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Judge whether a player is trained enough for snapshot. If yes, call player's ``snapshot``, create a\n            historical player(prepare the checkpoint and add it to the shared payoff), then mutate it, and return True.\n            Otherwise, return False.\n        Arguments:\n            - player_id (:obj:`ActivePlayer`): The active player's id.\n        Returns:\n            - snapshot_or_not (:obj:`dict`): Whether the active player is snapshotted.\n        "
        with self._active_players_lock:
            idx = self.active_players_ids.index(player_id)
            player = self.active_players[idx]
            if force or player.is_trained_enough():
                hp = player.snapshot(self.metric_env)
                self.save_checkpoint(player.checkpoint_path, hp.checkpoint_path)
                self.historical_players.append(hp)
                self.payoff.add_player(hp)
                self._mutate_player(player)
                return True
            else:
                return False

    @abstractmethod
    def _mutate_player(self, player: ActivePlayer) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Players have the probability to mutate, e.g. Reset network parameters.\n            Called by ``self.judge_snapshot``.\n        Arguments:\n            - player (:obj:`ActivePlayer`): The active player that may mutate.\n        '
        raise NotImplementedError

    def update_active_player(self, player_info: dict) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Update an active player's info.\n        Arguments:\n            - player_info (:obj:`dict`): Info dict of the player which is to be updated.\n        ArgumentsKeys:\n            - necessary: `player_id`, `train_iteration`\n        "
        try:
            idx = self.active_players_ids.index(player_info['player_id'])
            player = self.active_players[idx]
            return self._update_player(player, player_info)
        except ValueError as e:
            print(e)

    @abstractmethod
    def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Update an active player. Called by ``self.update_active_player``.\n        Arguments:\n            - player (:obj:`ActivePlayer`): The active player that will be updated.\n            - player_info (:obj:`dict`): Info dict of the active player which is to be updated.\n        '
        raise NotImplementedError

    def finish_job(self, job_info: dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Finish current job. Update shared payoff to record the game results.\n        Arguments:\n            - job_info (:obj:`dict`): A dict containing job result information.\n        '
        self.payoff.update(job_info)
        if 'eval_flag' in job_info and job_info['eval_flag']:
            (home_id, away_id) = job_info['player_id']
            (home_player, away_player) = (self.get_player_by_id(home_id), self.get_player_by_id(away_id))
            job_info_result = job_info['result']
            if isinstance(job_info_result[0], list):
                job_info_result = sum(job_info_result, [])
            (home_player.rating, away_player.rating) = self.metric_env.rate_1vs1(home_player.rating, away_player.rating, result=job_info_result)

    def get_player_by_id(self, player_id: str) -> 'Player':
        if False:
            for i in range(10):
                print('nop')
        if 'historical' in player_id:
            return [p for p in self.historical_players if p.player_id == player_id][0]
        else:
            return [p for p in self.active_players if p.player_id == player_id][0]

    @staticmethod
    def save_checkpoint(src_checkpoint, dst_checkpoint) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Copy a checkpoint from path ``src_checkpoint`` to path ``dst_checkpoint``.\n        Arguments:\n            - src_checkpoint (:obj:`str`): Source checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth\n            - dst_checkpoint (:obj:`str`): Destination checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth\n        "
        checkpoint = read_file(src_checkpoint)
        save_file(dst_checkpoint, checkpoint)

    def player_rank(self, string: bool=False) -> Union[str, Dict[str, float]]:
        if False:
            for i in range(10):
                print('nop')
        rank = {}
        for p in self.active_players + self.historical_players:
            name = p.player_id
            rank[name] = p.rating.exposure
        if string:
            headers = ['Player ID', 'Rank (TrueSkill)']
            data = []
            for (k, v) in rank.items():
                data.append([k, '{:.2f}'.format(v)])
            s = '\n' + tabulate(data, headers=headers, tablefmt='pipe')
            return s
        else:
            return rank

def create_league(cfg: EasyDict, *args) -> BaseLeague:
    if False:
        i = 10
        return i + 15
    "\n    Overview:\n        Given the key (league_type), create a new league instance if in league_mapping's values,\n        or raise an KeyError. In other words, a derived league must first register then call ``create_league``\n        to get the instance object.\n    Arguments:\n        - cfg (:obj:`EasyDict`): league config, necessary keys: [league.import_module, league.learner_type]\n    Returns:\n        - league (:obj:`BaseLeague`): the created new league, should be an instance of one of             league_mapping's values\n    "
    import_module(cfg.get('import_names', []))
    return LEAGUE_REGISTRY.build(cfg.league_type, *args, cfg=cfg)