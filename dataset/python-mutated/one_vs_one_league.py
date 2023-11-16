from easydict import EasyDict
from typing import Optional
from ding.utils import LEAGUE_REGISTRY
from .base_league import BaseLeague
from .player import ActivePlayer

@LEAGUE_REGISTRY.register('one_vs_one')
class OneVsOneLeague(BaseLeague):
    """
    Overview:
        One vs One battle game league.
        Decide which two players will play against each other.
    Interface:
        __init__, run, close, finish_job, update_active_player
    """
    config = dict(league_type='one_vs_one', import_names=['ding.league'], player_category=['default'], active_players=dict(naive_sp_player=1), naive_sp_player=dict(one_phase_step=10, branch_probs=dict(pfsp=0.5, sp=0.5), strong_win_rate=0.7), use_pretrain=False, use_pretrain_init_historical=False, pretrain_checkpoint_path=dict(default='default_cate_pretrain.pth'), payoff=dict(type='battle', decay=0.99, min_win_rate_games=8), metric=dict(mu=0, sigma=25 / 3, beta=25 / 3 / 2, tau=0.0, draw_probability=0.02))

    def _get_job_info(self, player: ActivePlayer, eval_flag: bool=False) -> dict:
        if False:
            return 10
        "\n        Overview:\n            Get player's job related info, called by ``_launch_job``.\n        Arguments:\n            - player (:obj:`ActivePlayer`): The active player that will be assigned a job.\n        "
        assert isinstance(player, ActivePlayer), player.__class__
        player_job_info = EasyDict(player.get_job(eval_flag))
        if eval_flag:
            return {'agent_num': 1, 'launch_player': player.player_id, 'player_id': [player.player_id], 'checkpoint_path': [player.checkpoint_path], 'player_active_flag': [isinstance(player, ActivePlayer)], 'eval_opponent': player_job_info.opponent}
        else:
            return {'agent_num': 2, 'launch_player': player.player_id, 'player_id': [player.player_id, player_job_info.opponent.player_id], 'checkpoint_path': [player.checkpoint_path, player_job_info.opponent.checkpoint_path], 'player_active_flag': [isinstance(p, ActivePlayer) for p in [player, player_job_info.opponent]]}

    def _mutate_player(self, player: ActivePlayer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Players have the probability to be reset to supervised learning model parameters.\n        Arguments:\n            - player (:obj:`ActivePlayer`): The active player that may mutate.\n        '
        pass

    def _update_player(self, player: ActivePlayer, player_info: dict) -> Optional[bool]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Update an active player, called by ``self.update_active_player``.\n        Arguments:\n            - player (:obj:`ActivePlayer`): The active player that will be updated.\n            - player_info (:obj:`dict`): An info dict of the active player which is to be updated.\n        Returns:\n            - increment_eval_difficulty (:obj:`bool`): Only return this when evaluator calls this method.                 Return True if difficulty is incremented; Otherwise return False (difficulty will not increment                 when it is already the most difficult or evaluator loses)\n        '
        assert isinstance(player, ActivePlayer)
        if 'train_iteration' in player_info:
            player.total_agent_step = player_info['train_iteration']
            return False
        elif 'eval_win' in player_info:
            if player_info['eval_win']:
                increment_eval_difficulty = player.increment_eval_difficulty()
                return increment_eval_difficulty
            else:
                return False