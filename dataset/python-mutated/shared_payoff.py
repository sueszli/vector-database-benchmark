import copy
from collections import defaultdict
from typing import Tuple, Optional
from easydict import EasyDict
from tabulate import tabulate
import numpy as np
from ding.utils import LockContext, LockContextType
from .player import Player

class BattleRecordDict(dict):
    """
    Overview:
        A dict which is used to record battle game result.
        Initialized four fixed keys: `wins`, `draws`, `losses`, `games`; Each with value 0.
    Interfaces:
        __mul__
    """
    data_keys = ['wins', 'draws', 'losses', 'games']

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Initialize four fixed keys ['wins', 'draws', 'losses', 'games'] and set value to 0\n        "
        super(BattleRecordDict, self).__init__()
        for k in self.data_keys:
            self[k] = 0

    def __mul__(self, decay: float) -> dict:
        if False:
            return 10
        "\n        Overview:\n            Multiply each key's value with the input multiplier ``decay``\n        Arguments:\n            - decay (:obj:`float`): The multiplier.\n        Returns:\n            - obj (:obj:`dict`): A deepcopied RecordDict after multiplication decay.\n        "
        obj = copy.deepcopy(self)
        for k in obj.keys():
            obj[k] *= decay
        return obj

class BattleSharedPayoff:
    """
    Overview:
        Payoff data structure to record historical match result, this payoff is shared among all the players.
        Use LockContext to ensure thread safe, since all players from all threads can access and modify it.
    Interface:
        __getitem__, add_player, update, get_key
    Property:
        players
    """

    def __init__(self, cfg: EasyDict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Initialize battle payoff\n        Arguments:\n            - cfg (:obj:`dict`): config(contains {decay, min_win_rate_games})\n        '
        self._players = []
        self._players_ids = []
        self._data = defaultdict(BattleRecordDict)
        self._decay = cfg.decay
        self._min_win_rate_games = cfg.get('min_win_rate_games', 8)
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        headers = ['Home Player', 'Away Player', 'Wins', 'Draws', 'Losses', 'Naive Win Rate']
        data = []
        for (k, v) in self._data.items():
            k1 = k.split('-')
            if 'historical' in k1[0]:
                naive_win_rate = (v['losses'] + v['draws'] / 2) / (v['wins'] + v['losses'] + v['draws'] + 1e-08)
                data.append([k1[1], k1[0], v['losses'], v['draws'], v['wins'], naive_win_rate])
            else:
                naive_win_rate = (v['wins'] + v['draws'] / 2) / (v['wins'] + v['losses'] + v['draws'] + 1e-08)
                data.append([k1[0], k1[1], v['wins'], v['draws'], v['losses'], naive_win_rate])
        data = sorted(data, key=lambda x: x[0])
        s = tabulate(data, headers=headers, tablefmt='pipe')
        return s

    def __getitem__(self, players: tuple) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Get win rates between home players and away players one by one\n        Arguments:\n            - players (:obj:`tuple`): A tuple of (home, away), each one is a player or a player list.\n        Returns:\n            - win_rates (:obj:`np.ndarray`): Win rate (squeezed, see Shape for more details)                 between each player from home and each player from away.\n        Shape:\n            - win_rates: Assume there are m home players and n away players.(m,n > 0)\n\n                - m != 1 and n != 1: shape is (m, n)\n                - m == 1: shape is (n)\n                - n == 1: shape is (m)\n        '
        with self._lock:
            (home, away) = players
            assert isinstance(home, list) or isinstance(home, Player)
            assert isinstance(away, list) or isinstance(away, Player)
            if isinstance(home, Player):
                home = [home]
            if isinstance(away, Player):
                away = [away]
            win_rates = np.array([[self._win_rate(h.player_id, a.player_id) for a in away] for h in home])
            if len(home) == 1 or len(away) == 1:
                win_rates = win_rates.reshape(-1)
            return win_rates

    def _win_rate(self, home: str, away: str) -> float:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Calculate win rate of one `home player` vs one `away player`\n        Arguments:\n            - home (:obj:`str`): home player id to access win rate\n            - away (:obj:`str`): away player id to access win rate\n        Returns:\n            - win rate (:obj:`float`): float win rate value.                 Only when total games is no less than ``self._min_win_rate_games``,                 can the win rate be calculated by (wins + draws/2) / games, or return 0.5 by default.\n        '
        (key, reverse) = self.get_key(home, away)
        handle = self._data[key]
        if handle['games'] < self._min_win_rate_games:
            return 0.5
        wins = handle['wins'] if not reverse else handle['losses']
        return (wins + 0.5 * handle['draws']) / handle['games']

    @property
    def players(self):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Get all the players\n        Returns:\n            - players (:obj:`list`): players list\n        '
        with self._lock:
            return self._players

    def add_player(self, player: Player) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Add a player to the shared payoff.\n        Arguments:\n            - player (:obj:`Player`): The player to be added. Usually is a new one to the league as well.\n        '
        with self._lock:
            self._players.append(player)
            self._players_ids.append(player.player_id)

    def update(self, job_info: dict) -> bool:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Update payoff with job_info when a job is to be finished.\n            If update succeeds, return True; If raises an exception when updating, resolve it and return False.\n        Arguments:\n            - job_info (:obj:`dict`): A dict containing job result information.\n        Returns:\n            - result (:obj:`bool`): Whether update is successful.\n\n        .. note::\n            job_info has at least 5 keys ['launch_player', 'player_id', 'env_num', 'episode_num', 'result'].\n            Key ``player_id`` 's value is a tuple of (home_id, away_id).\n            Key ``result`` 's value is a two-layer list with the length of (episode_num, env_num).\n        "

        def _win_loss_reverse(result_: str, reverse_: bool) -> str:
            if False:
                for i in range(10):
                    print('nop')
            if result_ == 'draws' or not reverse_:
                return result_
            reverse_dict = {'wins': 'losses', 'losses': 'wins'}
            return reverse_dict[result_]
        with self._lock:
            (home_id, away_id) = job_info['player_id']
            job_info_result = job_info['result']
            if not isinstance(job_info_result[0], list):
                job_info_result = [job_info_result]
            try:
                assert home_id in self._players_ids, 'home_id error'
                assert away_id in self._players_ids, 'away_id error'
                assert all([i in BattleRecordDict.data_keys[:3] for j in job_info_result for i in j]), 'results error'
            except Exception as e:
                print('[ERROR] invalid job_info: {}\n\tError reason is: {}'.format(job_info, e))
                return False
            if home_id == away_id:
                (key, reverse) = self.get_key(home_id, away_id)
                self._data[key]['draws'] += 1
                self._data[key]['games'] += 1
            else:
                (key, reverse) = self.get_key(home_id, away_id)
                for one_episode_result in job_info_result:
                    for one_episode_result_per_env in one_episode_result:
                        self._data[key] *= self._decay
                        self._data[key]['games'] += 1
                        result = _win_loss_reverse(one_episode_result_per_env, reverse)
                        self._data[key][result] += 1
            return True

    def get_key(self, home: str, away: str) -> Tuple[str, bool]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Join home player id and away player id in alphabetival order.\n        Arguments:\n            - home (:obj:`str`): Home player id\n            - away (:obj:`str`): Away player id\n        Returns:\n            - key (:obj:`str`): Tow ids sorted in alphabetical order, and joined by '-'.\n            - reverse (:obj:`bool`): Whether the two player ids are reordered.\n        "
        assert isinstance(home, str)
        assert isinstance(away, str)
        reverse = False
        if home <= away:
            tmp = [home, away]
        else:
            tmp = [away, home]
            reverse = True
        return ('-'.join(tmp), reverse)

def create_payoff(cfg: EasyDict) -> Optional[BattleSharedPayoff]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Overview:\n        Given the key (payoff type), now supports keys ['solo', 'battle'],\n        create a new payoff instance if in payoff_mapping's values, or raise an KeyError.\n    Arguments:\n        - cfg (:obj:`EasyDict`): payoff config containing at least one key 'type'\n    Returns:\n        - payoff (:obj:`BattleSharedPayoff` or :obj:`SoloSharedPayoff`): the created new payoff,             should be an instance of one of payoff_mapping's values\n    "
    payoff_mapping = {'battle': BattleSharedPayoff}
    payoff_type = cfg.type
    if payoff_type not in payoff_mapping.keys():
        raise KeyError('not support payoff type: {}'.format(payoff_type))
    else:
        return payoff_mapping[payoff_type](cfg)