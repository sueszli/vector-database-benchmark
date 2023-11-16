import copy
import importlib
import logging
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
import torch.multiprocessing
from pandas import DataFrame
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import is_masking_supported
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv
from freqtrade.freqai.RL.BaseEnvironment import BaseActions, BaseEnvironment, Positions
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback
from freqtrade.persistence import Trade
logger = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')
SB3_MODELS = ['PPO', 'A2C', 'DQN']
SB3_CONTRIB_MODELS = ['TRPO', 'ARS', 'RecurrentPPO', 'MaskablePPO', 'QRDQN']

class BaseReinforcementLearningModel(IFreqaiModel):
    """
    User created Reinforcement Learning Model prediction class
    """

    def __init__(self, **kwargs) -> None:
        if False:
            return 10
        super().__init__(config=kwargs['config'])
        self.max_threads = min(self.freqai_info['rl_config'].get('cpu_count', 1), max(int(self.max_system_threads / 2), 1))
        th.set_num_threads(self.max_threads)
        self.reward_params = self.freqai_info['rl_config']['model_reward_parameters']
        self.train_env: Union[VecMonitor, SubprocVecEnv, gym.Env] = gym.Env()
        self.eval_env: Union[VecMonitor, SubprocVecEnv, gym.Env] = gym.Env()
        self.eval_callback: Optional[MaskableEvalCallback] = None
        self.model_type = self.freqai_info['rl_config']['model_type']
        self.rl_config = self.freqai_info['rl_config']
        self.df_raw: DataFrame = DataFrame()
        self.continual_learning = self.freqai_info.get('continual_learning', False)
        if self.model_type in SB3_MODELS:
            import_str = 'stable_baselines3'
        elif self.model_type in SB3_CONTRIB_MODELS:
            import_str = 'sb3_contrib'
        else:
            raise OperationalException(f'{self.model_type} not available in stable_baselines3 or sb3_contrib. please choose one of {SB3_MODELS} or {SB3_CONTRIB_MODELS}')
        mod = importlib.import_module(import_str, self.model_type)
        self.MODELCLASS = getattr(mod, self.model_type)
        self.policy_type = self.freqai_info['rl_config']['policy_type']
        self.unset_outlier_removal()
        self.net_arch = self.rl_config.get('net_arch', [128, 128])
        self.dd.model_type = import_str
        self.tensorboard_callback: TensorboardCallback = TensorboardCallback(verbose=1, actions=BaseActions)

    def unset_outlier_removal(self):
        if False:
            print('Hello World!')
        '\n        If user has activated any function that may remove training points, this\n        function will set them to false and warn them\n        '
        if self.ft_params.get('use_SVM_to_remove_outliers', False):
            self.ft_params.update({'use_SVM_to_remove_outliers': False})
            logger.warning('User tried to use SVM with RL. Deactivating SVM.')
        if self.ft_params.get('use_DBSCAN_to_remove_outliers', False):
            self.ft_params.update({'use_DBSCAN_to_remove_outliers': False})
            logger.warning('User tried to use DBSCAN with RL. Deactivating DBSCAN.')
        if self.ft_params.get('DI_threshold', False):
            self.ft_params.update({'DI_threshold': False})
            logger.warning('User tried to use DI_threshold with RL. Deactivating DI_threshold.')
        if self.freqai_info['data_split_parameters'].get('shuffle', False):
            self.freqai_info['data_split_parameters'].update({'shuffle': False})
            logger.warning('User tried to shuffle training data. Setting shuffle to False')

    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Filter the training data and train a model to it. Train makes heavy use of the datakitchen\n        for storing, saving, loading, and analyzing the data.\n        :param unfiltered_df: Full dataframe for the current training period\n        :param metadata: pair metadata from strategy.\n        :returns:\n        :model: Trained model which can be used to inference (self.predict)\n        '
        logger.info(f'--------------------Starting training {pair} --------------------')
        (features_filtered, labels_filtered) = dk.filter_features(unfiltered_df, dk.training_features_list, dk.label_list, training_filter=True)
        dd: Dict[str, Any] = dk.make_train_test_datasets(features_filtered, labels_filtered)
        self.df_raw = copy.deepcopy(dd['train_features'])
        dk.fit_labels()
        (prices_train, prices_test) = self.build_ohlc_price_dataframes(dk.data_dictionary, pair, dk)
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        (dd['train_features'], dd['train_labels'], dd['train_weights']) = dk.feature_pipeline.fit_transform(dd['train_features'], dd['train_labels'], dd['train_weights'])
        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (dd['test_features'], dd['test_labels'], dd['test_weights']) = dk.feature_pipeline.transform(dd['test_features'], dd['test_labels'], dd['test_weights'])
        logger.info(f"Training model on {len(dk.data_dictionary['train_features'].columns)} features and {len(dd['train_features'])} data points")
        self.set_train_and_eval_environments(dd, prices_train, prices_test, dk)
        model = self.fit(dd, dk)
        logger.info(f'--------------------done training {pair}--------------------')
        return model

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, DataFrame], prices_train: DataFrame, prices_test: DataFrame, dk: FreqaiDataKitchen):
        if False:
            for i in range(10):
                print('nop')
        '\n        User can override this if they are using a custom MyRLEnv\n        :param data_dictionary: dict = common data dictionary containing train and test\n            features/labels/weights.\n        :param prices_train/test: DataFrame = dataframe comprised of the prices to be used in the\n            environment during training or testing\n        :param dk: FreqaiDataKitchen = the datakitchen for the current pair\n        '
        train_df = data_dictionary['train_features']
        test_df = data_dictionary['test_features']
        env_info = self.pack_env_dict(dk.pair)
        self.train_env = self.MyRLEnv(df=train_df, prices=prices_train, **env_info)
        self.eval_env = Monitor(self.MyRLEnv(df=test_df, prices=prices_test, **env_info))
        self.eval_callback = MaskableEvalCallback(self.eval_env, deterministic=True, render=False, eval_freq=len(train_df), best_model_save_path=str(dk.data_path), use_masking=self.model_type == 'MaskablePPO' and is_masking_supported(self.eval_env))
        actions = self.train_env.get_actions()
        self.tensorboard_callback = TensorboardCallback(verbose=1, actions=actions)

    def pack_env_dict(self, pair: str) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Create dictionary of environment arguments\n        '
        env_info = {'window_size': self.CONV_WIDTH, 'reward_kwargs': self.reward_params, 'config': self.config, 'live': self.live, 'can_short': self.can_short, 'pair': pair, 'df_raw': self.df_raw}
        if self.data_provider:
            env_info['fee'] = self.data_provider._exchange.get_fee(symbol=self.data_provider.current_whitelist()[0])
        return env_info

    @abstractmethod
    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        if False:
            return 10
        '\n        Agent customizations and abstract Reinforcement Learning customizations\n        go in here. Abstract method, so this function must be overridden by\n        user class.\n        '
        return

    def get_state_info(self, pair: str) -> Tuple[float, float, int]:
        if False:
            return 10
        '\n        State info during dry/live (not backtesting) which is fed back\n        into the model.\n        :param pair: str = COIN/STAKE to get the environment information for\n        :return:\n        :market_side: float = representing short, long, or neutral for\n            pair\n        :current_profit: float = unrealized profit of the current trade\n        :trade_duration: int = the number of candles that the trade has\n            been open for\n        '
        open_trades = Trade.get_trades_proxy(is_open=True)
        market_side = 0.5
        current_profit: float = 0
        trade_duration = 0
        for trade in open_trades:
            if trade.pair == pair:
                if self.data_provider._exchange is None:
                    logger.error('No exchange available.')
                    return (0, 0, 0)
                else:
                    current_rate = self.data_provider._exchange.get_rate(pair, refresh=False, side='exit', is_short=trade.is_short)
                now = datetime.now(timezone.utc).timestamp()
                trade_duration = int((now - trade.open_date_utc.timestamp()) / self.base_tf_seconds)
                current_profit = trade.calc_profit_ratio(current_rate)
                if trade.is_short:
                    market_side = 0
                else:
                    market_side = 1
        return (market_side, current_profit, int(trade_duration))

    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Filter the prediction features data and predict with it.\n        :param unfiltered_dataframe: Full dataframe for the current backtest period.\n        :return:\n        :pred_df: dataframe containing the predictions\n        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove\n        data (NaNs) or felt uncertain about data (PCA and DI index)\n        '
        dk.find_features(unfiltered_df)
        (filtered_dataframe, _) = dk.filter_features(unfiltered_df, dk.training_features_list, training_filter=False)
        dk.data_dictionary['prediction_features'] = self.drop_ohlc_from_df(filtered_dataframe, dk)
        (dk.data_dictionary['prediction_features'], _, _) = dk.feature_pipeline.transform(dk.data_dictionary['prediction_features'], outlier_check=True)
        pred_df = self.rl_model_predict(dk.data_dictionary['prediction_features'], dk, self.model)
        pred_df.fillna(0, inplace=True)
        return (pred_df, dk.do_predict)

    def rl_model_predict(self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any) -> DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        A helper function to make predictions in the Reinforcement learning module.\n        :param dataframe: DataFrame = the dataframe of features to make the predictions on\n        :param dk: FreqaiDatakitchen = data kitchen for the current pair\n        :param model: Any = the trained model used to inference the features.\n        '
        output = pd.DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)

        def _predict(window):
            if False:
                print('Hello World!')
            observations = dataframe.iloc[window.index]
            if self.live and self.rl_config.get('add_state_info', False):
                (market_side, current_profit, trade_duration) = self.get_state_info(dk.pair)
                observations['current_profit_pct'] = current_profit
                observations['position'] = market_side
                observations['trade_duration'] = trade_duration
            (res, _) = model.predict(observations, deterministic=True)
            return res
        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)
        return output

    def build_ohlc_price_dataframes(self, data_dictionary: dict, pair: str, dk: FreqaiDataKitchen) -> Tuple[DataFrame, DataFrame]:
        if False:
            i = 10
            return i + 15
        '\n        Builds the train prices and test prices for the environment.\n        '
        pair = pair.replace(':', '')
        train_df = data_dictionary['train_features']
        test_df = data_dictionary['test_features']
        tf = self.config['timeframe']
        rename_dict = {'%-raw_open': 'open', '%-raw_low': 'low', '%-raw_high': ' high', '%-raw_close': 'close'}
        rename_dict_old = {f'%-{pair}raw_open_{tf}': 'open', f'%-{pair}raw_low_{tf}': 'low', f'%-{pair}raw_high_{tf}': ' high', f'%-{pair}raw_close_{tf}': 'close'}
        prices_train = train_df.filter(rename_dict.keys(), axis=1)
        prices_train_old = train_df.filter(rename_dict_old.keys(), axis=1)
        if prices_train.empty or not prices_train_old.empty:
            if not prices_train_old.empty:
                prices_train = prices_train_old
                rename_dict = rename_dict_old
            logger.warning('Reinforcement learning module didnt find the correct raw prices assigned in feature_engineering_standard(). Please assign them with:\ndataframe["%-raw_close"] = dataframe["close"]\ndataframe["%-raw_open"] = dataframe["open"]\ndataframe["%-raw_high"] = dataframe["high"]\ndataframe["%-raw_low"] = dataframe["low"]\ninside `feature_engineering_standard()')
        elif prices_train.empty:
            raise OperationalException('No prices found, please follow log warning instructions to correct the strategy.')
        prices_train.rename(columns=rename_dict, inplace=True)
        prices_train.reset_index(drop=True)
        prices_test = test_df.filter(rename_dict.keys(), axis=1)
        prices_test.rename(columns=rename_dict, inplace=True)
        prices_test.reset_index(drop=True)
        train_df = self.drop_ohlc_from_df(train_df, dk)
        test_df = self.drop_ohlc_from_df(test_df, dk)
        return (prices_train, prices_test)

    def drop_ohlc_from_df(self, df: DataFrame, dk: FreqaiDataKitchen):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a dataframe, drop the ohlc data\n        '
        drop_list = ['%-raw_open', '%-raw_low', '%-raw_high', '%-raw_close']
        if self.rl_config['drop_ohlc_from_features']:
            df.drop(drop_list, axis=1, inplace=True)
            feature_list = dk.training_features_list
            dk.training_features_list = [e for e in feature_list if e not in drop_list]
        return df

    def load_model_from_disk(self, dk: FreqaiDataKitchen) -> Any:
        if False:
            print('Hello World!')
        '\n        Can be used by user if they are trying to limit_ram_usage *and*\n        perform continual learning.\n        For now, this is unused.\n        '
        exists = Path(dk.data_path / f'{dk.model_filename}_model').is_file()
        if exists:
            model = self.MODELCLASS.load(dk.data_path / f'{dk.model_filename}_model')
        else:
            logger.info('No model file on disk to continue learning from.')
        return model

    def _on_stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Hook called on bot shutdown. Close SubprocVecEnv subprocesses for clean shutdown.\n        '
        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()

    class MyRLEnv(Base5ActionRLEnv):
        """
        User can override any function in BaseRLEnv and gym.Env. Here the user
        sets a custom reward based on profit and trade duration.
        """

        def calculate_reward(self, action: int) -> float:
            if False:
                return 10
            '\n            An example reward function. This is the one function that users will likely\n            wish to inject their own creativity into.\n\n            Warning!\n            This is function is a showcase of functionality designed to show as many possible\n            environment control features as possible. It is also designed to run quickly\n            on small computers. This is a benchmark, it is *not* for live production.\n\n            :param action: int = The action made by the agent for the current candle.\n            :return:\n            float = the reward to give to the agent for current step (used for optimization\n                of weights in NN)\n            '
            if not self._is_valid(action):
                return -2
            pnl = self.get_unrealized_profit()
            factor = 100.0
            rsi_now = self.raw_features[f"%-rsi-period-10_shift-1_{self.pair}_{self.config['timeframe']}"].iloc[self._current_tick]
            if action in (Actions.Long_enter.value, Actions.Short_enter.value) and self._position == Positions.Neutral:
                if rsi_now < 40:
                    factor = 40 / rsi_now
                else:
                    factor = 1
                return 25 * factor
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1
            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
            if self._last_trade_tick:
                trade_duration = self._current_tick - self._last_trade_tick
            else:
                trade_duration = 0
            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5
            if self._position in (Positions.Short, Positions.Long) and action == Actions.Neutral.value:
                return -1 * trade_duration / max_trade_duration
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)
            if action == Actions.Short_exit.value and self._position == Positions.Short:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)
            return 0.0

def make_env(MyRLEnv: Type[BaseEnvironment], env_id: str, rank: int, seed: int, train_df: DataFrame, price: DataFrame, env_info: Dict[str, Any]={}) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Utility function for multiprocessed env.\n\n    :param env_id: (str) the environment ID\n    :param num_env: (int) the number of environment you wish to have in subprocesses\n    :param seed: (int) the inital seed for RNG\n    :param rank: (int) index of the subprocess\n    :param env_info: (dict) all required arguments to instantiate the environment.\n    :return: (Callable)\n    '

    def _init() -> gym.Env:
        if False:
            i = 10
            return i + 15
        env = MyRLEnv(df=train_df, prices=price, id=env_id, seed=seed + rank, **env_info)
        return env
    set_random_seed(seed)
    return _init