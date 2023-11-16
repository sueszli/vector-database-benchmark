from typing import Optional, Union
import numpy as np
from ding.utils import PLAYER_REGISTRY
from .player import ActivePlayer, HistoricalPlayer
from .algorithm import pfsp

@PLAYER_REGISTRY.register('main_player')
class MainPlayer(ActivePlayer):
    """
    Overview:
        Main player in league training.
        Default branch (0.5 pfsp, 0.35 sp, 0.15 veri).
        Default snapshot every 2e9 steps.
        Default mutate prob = 0 (never mutate).
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_iteration
    """
    _name = 'MainPlayer'

    def _pfsp_branch(self) -> HistoricalPlayer:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Select prioritized fictitious self-play opponent, should be a historical player.\n        Returns:\n            - player (:obj:`HistoricalPlayer`): the selected historical player\n        '
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer))
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='squared')
        return self._get_opponent(historical, p)

    def _sp_branch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Select normal self-play opponent\n        '
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)
        if self._payoff[self, main_opponent] > 1 - self._strong_win_rate:
            return main_opponent
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id)
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='variance')
        return self._get_opponent(historical, p)

    def _verification_branch(self):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Verify no strong historical main exploiter and no forgotten historical past main player\n        '
        main_exploiters = self._get_players(lambda p: isinstance(p, MainExploiter))
        exp_historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer) and any([p.parent_id == m.player_id for m in main_exploiters]))
        win_rates = self._payoff[self, exp_historical]
        if len(win_rates) and win_rates.min() < 1 - self._strong_win_rate:
            p = pfsp(win_rates, weighting='squared')
            return self._get_opponent(exp_historical, p)
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)
        main_historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id)
        win_rates = self._payoff[self, main_historical]
        if len(win_rates) and win_rates.min() < self._strong_win_rate:
            p = pfsp(win_rates, weighting='squared')
            return self._get_opponent(main_historical, p)
        return self._sp_branch()

    def is_trained_enough(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, HistoricalPlayer))

    def mutate(self, info: dict) -> None:
        if False:
            return 10
        '\n        Overview:\n            MainPlayer does not mutate\n        '
        pass

@PLAYER_REGISTRY.register('main_exploiter')
class MainExploiter(ActivePlayer):
    """
    Overview:
        Main exploiter in league training. Can identify weaknesses of main agents, and consequently make them
        more robust.
        Default branch (1.0 main_players).
        Default snapshot when defeating all 3 main players in the league in more than 70% of games,
        or timeout of 4e9 steps.
        Default mutate prob = 1 (must mutate).
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_iteration
    """
    _name = 'MainExploiter'

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Initialize ``min_valid_win_rate`` additionally\n        Note:\n            - min_valid_win_rate (:obj:`float`): only when win rate against the main player is greater than this,                 can the main player be regarded as able to produce valid training signals to be selected\n        '
        super(MainExploiter, self).__init__(*args, **kwargs)
        self._min_valid_win_rate = self._cfg.min_valid_win_rate

    def _main_players_branch(self):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Select main player or historical player snapshot from main player as opponent\n        Returns:\n            - player (:obj:`Player`): the selected main player (active/historical)\n        '
        main_players = self._get_players(lambda p: isinstance(p, MainPlayer))
        main_opponent = self._get_opponent(main_players)
        if self._payoff[self, main_opponent] >= self._min_valid_win_rate:
            return main_opponent
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer) and p.parent_id == main_opponent.player_id)
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='variance')
        return self._get_opponent(historical, p)

    def is_trained_enough(self):
        if False:
            while True:
                i = 10
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, MainPlayer))

    def mutate(self, info: dict) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Main exploiter is sure to mutate(reset) to the supervised learning player\n        Returns:\n            - mutate_ckpt_path (:obj:`str`): mutation target checkpoint path\n        '
        return info['reset_checkpoint_path']

@PLAYER_REGISTRY.register('league_exploiter')
class LeagueExploiter(ActivePlayer):
    """
    Overview:
        League exploiter in league training. Can identify global blind spots in the league (strategies that no player
        in the league can beat, but that are not necessarily robust themselves).
        Default branch (1.0 pfsp).
        Default snapshot when defeating all players in the league in more than 70% of games, or timeout of 2e9 steps.
        Default mutate prob = 0.25.
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, train_iteration
    """
    _name = 'LeagueExploiter'

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Initialize ``mutate_prob`` additionally\n        Note:\n            - mutate_prob (:obj:`float`): the mutation probability of league exploiter. should be in [0, 1]\n        '
        super(LeagueExploiter, self).__init__(*args, **kwargs)
        assert 0 <= self._cfg.mutate_prob <= 1
        self.mutate_prob = self._cfg.mutate_prob

    def _pfsp_branch(self) -> HistoricalPlayer:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Select prioritized fictitious self-play opponent\n        Returns:\n            - player (:obj:`HistoricalPlayer`): the selected historical player\n        Note:\n            This branch is the same as the psfp branch in MainPlayer\n        '
        historical = self._get_players(lambda p: isinstance(p, HistoricalPlayer))
        win_rates = self._payoff[self, historical]
        p = pfsp(win_rates, weighting='squared')
        return self._get_opponent(historical, p)

    def is_trained_enough(self) -> bool:
        if False:
            while True:
                i = 10
        return super().is_trained_enough(select_fn=lambda p: isinstance(p, HistoricalPlayer))

    def mutate(self, info) -> Union[str, None]:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            League exploiter can mutate to the supervised learning player with 0.25 prob\n        Returns:\n            - ckpt_path (:obj:`Union[str, None]`): with ``mutate_prob`` prob returns the pretrained model's ckpt path,                 with left 1 - ``mutate_prob`` prob returns None, which means no mutation\n        "
        p = np.random.uniform()
        if p < self.mutate_prob:
            return info['reset_checkpoint_path']
        return None