import logging
import random
from abc import abstractmethod
from enum import Enum
from typing import List, Optional, Type, Union
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from pandas import DataFrame
from freqtrade.exceptions import OperationalException
logger = logging.getLogger(__name__)

class BaseActions(Enum):
    """
    Default action space, mostly used for type handling.
    """
    Neutral = 0
    Long_enter = 1
    Long_exit = 2
    Short_enter = 3
    Short_exit = 4

class Positions(Enum):
    Short = 0
    Long = 1
    Neutral = 0.5

    def opposite(self):
        if False:
            for i in range(10):
                print('nop')
        return Positions.Short if self == Positions.Long else Positions.Long

class BaseEnvironment(gym.Env):
    """
    Base class for environments. This class is agnostic to action count.
    Inherited classes customize this to include varying action counts/types,
    See RL/Base5ActionRLEnv.py and RL/Base4ActionRLEnv.py
    """

    def __init__(self, df: DataFrame=DataFrame(), prices: DataFrame=DataFrame(), reward_kwargs: dict={}, window_size=10, starting_point=True, id: str='baseenv-1', seed: int=1, config: dict={}, live: bool=False, fee: float=0.0015, can_short: bool=False, pair: str='', df_raw: DataFrame=DataFrame()):
        if False:
            return 10
        '\n        Initializes the training/eval environment.\n        :param df: dataframe of features\n        :param prices: dataframe of prices to be used in the training environment\n        :param window_size: size of window (temporal) to pass to the agent\n        :param reward_kwargs: extra config settings assigned by user in `rl_config`\n        :param starting_point: start at edge of window or not\n        :param id: string id of the environment (used in backend for multiprocessed env)\n        :param seed: Sets the seed of the environment higher in the gym.Env object\n        :param config: Typical user configuration file\n        :param live: Whether or not this environment is active in dry/live/backtesting\n        :param fee: The fee to use for environmental interactions.\n        :param can_short: Whether or not the environment can short\n        '
        self.config: dict = config
        self.rl_config: dict = config['freqai']['rl_config']
        self.add_state_info: bool = self.rl_config.get('add_state_info', False)
        self.id: str = id
        self.max_drawdown: float = 1 - self.rl_config.get('max_training_drawdown_pct', 0.8)
        self.compound_trades: bool = config['stake_amount'] == 'unlimited'
        self.pair: str = pair
        self.raw_features: DataFrame = df_raw
        if self.config.get('fee', None) is not None:
            self.fee = self.config['fee']
        else:
            self.fee = fee
        self.actions: Type[Enum] = BaseActions
        self.tensorboard_metrics: dict = {}
        self.can_short: bool = can_short
        self.live: bool = live
        if not self.live and self.add_state_info:
            raise OperationalException('`add_state_info` is not available in backtesting. Change parameter to false in your rl_config. See `add_state_info` docs for more info.')
        self.seed(seed)
        self.reset_env(df, prices, window_size, reward_kwargs, starting_point)

    def reset_env(self, df: DataFrame, prices: DataFrame, window_size: int, reward_kwargs: dict, starting_point=True):
        if False:
            print('Hello World!')
        '\n        Resets the environment when the agent fails (in our case, if the drawdown\n        exceeds the user set max_training_drawdown_pct)\n        :param df: dataframe of features\n        :param prices: dataframe of prices to be used in the training environment\n        :param window_size: size of window (temporal) to pass to the agent\n        :param reward_kwargs: extra config settings assigned by user in `rl_config`\n        :param starting_point: start at edge of window or not\n        '
        self.signal_features: DataFrame = df
        self.prices: DataFrame = prices
        self.window_size: int = window_size
        self.starting_point: bool = starting_point
        self.rr: float = reward_kwargs['rr']
        self.profit_aim: float = reward_kwargs['profit_aim']
        if self.add_state_info:
            self.total_features = self.signal_features.shape[1] + 3
        else:
            self.total_features = self.signal_features.shape[1]
        self.shape = (window_size, self.total_features)
        self.set_action_space()
        self.observation_space = spaces.Box(low=-1, high=1, shape=self.shape, dtype=np.float32)
        self._start_tick: int = self.window_size
        self._end_tick: int = len(self.prices) - 1
        self._done: bool = False
        self._current_tick: int = self._start_tick
        self._last_trade_tick: Optional[int] = None
        self._position = Positions.Neutral
        self._position_history: list = [None]
        self.total_reward: float = 0
        self._total_profit: float = 1
        self._total_unrealized_profit: float = 1
        self.history: dict = {}
        self.trade_history: list = []

    def get_attr(self, attr: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the attribute of the environment\n        :param attr: attribute to return\n        :return: attribute\n        '
        return getattr(self, attr)

    @abstractmethod
    def set_action_space(self):
        if False:
            while True:
                i = 10
        '\n        Unique to the environment action count. Must be inherited.\n        '

    def action_masks(self) -> List[bool]:
        if False:
            for i in range(10):
                print('nop')
        return [self._is_valid(action.value) for action in self.actions]

    def seed(self, seed: int=1):
        if False:
            while True:
                i = 10
        (self.np_random, seed) = seeding.np_random(seed)
        return [seed]

    def tensorboard_log(self, metric: str, value: Optional[Union[int, float]]=None, inc: Optional[bool]=None, category: str='custom'):
        if False:
            i = 10
            return i + 15
        '\n        Function builds the tensorboard_metrics dictionary\n        to be parsed by the TensorboardCallback. This\n        function is designed for tracking incremented objects,\n        events, actions inside the training environment.\n        For example, a user can call this to track the\n        frequency of occurrence of an `is_valid` call in\n        their `calculate_reward()`:\n\n        def calculate_reward(self, action: int) -> float:\n            if not self._is_valid(action):\n                self.tensorboard_log("invalid")\n                return -2\n\n        :param metric: metric to be tracked and incremented\n        :param value: `metric` value\n        :param inc: (deprecated) sets whether the `value` is incremented or not\n        :param category: `metric` category\n        '
        increment = True if value is None else False
        value = 1 if increment else value
        if category not in self.tensorboard_metrics:
            self.tensorboard_metrics[category] = {}
        if not increment or metric not in self.tensorboard_metrics[category]:
            self.tensorboard_metrics[category][metric] = value
        else:
            self.tensorboard_metrics[category][metric] += value

    def reset_tensorboard_log(self):
        if False:
            print('Hello World!')
        self.tensorboard_metrics = {}

    def reset(self, seed=None):
        if False:
            print('Hello World!')
        '\n        Reset is called at the beginning of every episode\n        '
        self.reset_tensorboard_log()
        self._done = False
        if self.starting_point is True:
            if self.rl_config.get('randomize_starting_position', False):
                length_of_data = int(self._end_tick / 4)
                start_tick = random.randint(self.window_size + 1, length_of_data)
                self._start_tick = start_tick
            self._position_history = self._start_tick * [None] + [self._position]
        else:
            self._position_history = self.window_size * [None] + [self._position]
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Positions.Neutral
        self.total_reward = 0.0
        self._total_profit = 1.0
        self.history = {}
        self.trade_history = []
        self.portfolio_log_returns = np.zeros(len(self.prices))
        self._profits = [(self._start_tick, 1)]
        self.close_trade_profit = []
        self._total_unrealized_profit = 1
        return (self._get_observation(), self.history)

    @abstractmethod
    def step(self, action: int):
        if False:
            return 10
        '\n        Step depeneds on action types, this must be inherited.\n        '
        return

    def _get_observation(self):
        if False:
            i = 10
            return i + 15
        '\n        This may or may not be independent of action types, user can inherit\n        this in their custom "MyRLEnv"\n        '
        features_window = self.signal_features[self._current_tick - self.window_size:self._current_tick]
        if self.add_state_info:
            features_and_state = DataFrame(np.zeros((len(features_window), 3)), columns=['current_profit_pct', 'position', 'trade_duration'], index=features_window.index)
            features_and_state['current_profit_pct'] = self.get_unrealized_profit()
            features_and_state['position'] = self._position.value
            features_and_state['trade_duration'] = self.get_trade_duration()
            features_and_state = pd.concat([features_window, features_and_state], axis=1)
            return features_and_state
        else:
            return features_window

    def get_trade_duration(self):
        if False:
            return 10
        '\n        Get the trade duration if the agent is in a trade\n        '
        if self._last_trade_tick is None:
            return 0
        else:
            return self._current_tick - self._last_trade_tick

    def get_unrealized_profit(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the unrealized profit if the agent is in a trade\n        '
        if self._last_trade_tick is None:
            return 0.0
        if self._position == Positions.Neutral:
            return 0.0
        elif self._position == Positions.Short:
            current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
            return (last_trade_price - current_price) / last_trade_price
        elif self._position == Positions.Long:
            current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
            return (current_price - last_trade_price) / last_trade_price
        else:
            return 0.0

    @abstractmethod
    def is_tradesignal(self, action: int) -> bool:
        if False:
            print('Hello World!')
        '\n        Determine if the signal is a trade signal. This is\n        unique to the actions in the environment, and therefore must be\n        inherited.\n        '
        return True

    def _is_valid(self, action: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Determine if the signal is valid.This is\n        unique to the actions in the environment, and therefore must be\n        inherited.\n        '
        return True

    def add_entry_fee(self, price):
        if False:
            for i in range(10):
                print('nop')
        return price * (1 + self.fee)

    def add_exit_fee(self, price):
        if False:
            return 10
        return price / (1 + self.fee)

    def _update_history(self, info):
        if False:
            for i in range(10):
                print('nop')
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        for (key, value) in info.items():
            self.history[key].append(value)

    @abstractmethod
    def calculate_reward(self, action: int) -> float:
        if False:
            while True:
                i = 10
        '\n        An example reward function. This is the one function that users will likely\n        wish to inject their own creativity into.\n\n        Warning!\n        This is function is a showcase of functionality designed to show as many possible\n        environment control features as possible. It is also designed to run quickly\n        on small computers. This is a benchmark, it is *not* for live production.\n\n        :param action: int = The action made by the agent for the current candle.\n        :return:\n        float = the reward to give to the agent for current step (used for optimization\n            of weights in NN)\n        '

    def _update_unrealized_total_profit(self):
        if False:
            while True:
                i = 10
        '\n        Update the unrealized total profit incase of episode end.\n        '
        if self._position in (Positions.Long, Positions.Short):
            pnl = self.get_unrealized_profit()
            if self.compound_trades:
                unrl_profit = self._total_profit * (1 + pnl)
            else:
                unrl_profit = self._total_profit + pnl
            self._total_unrealized_profit = unrl_profit

    def _update_total_profit(self):
        if False:
            print('Hello World!')
        pnl = self.get_unrealized_profit()
        if self.compound_trades:
            self._total_profit = self._total_profit * (1 + pnl)
        else:
            self._total_profit += pnl

    def current_price(self) -> float:
        if False:
            i = 10
            return i + 15
        return self.prices.iloc[self._current_tick].open

    def get_actions(self) -> Type[Enum]:
        if False:
            while True:
                i = 10
        '\n        Used by SubprocVecEnv to get actions from\n        initialized env for tensorboard callback\n        '
        return self.actions