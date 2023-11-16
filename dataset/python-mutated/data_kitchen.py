import copy
import inspect
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
from datasieve.pipeline import Pipeline
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from freqtrade.configuration import TimeRange
from freqtrade.constants import DOCS_LINK, Config
from freqtrade.data.converter import reduce_dataframe_footprint
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy
SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600
logger = logging.getLogger(__name__)

class FreqaiDataKitchen:
    """
    Class designed to analyze data for a single pair. Employed by the IFreqaiModel class.
    Functionalities include holding, saving, loading, and analyzing the data.

    This object is not persistent, it is reinstantiated for each coin, each time the coin
    model needs to be inferenced or trained.

    Record of contribution:
    FreqAI was developed by a group of individuals who all contributed specific skillsets to the
    project.

    Conception and software development:
    Robert Caulk @robcaulk

    Theoretical brainstorming:
    Elin Törnquist @th0rntwig

    Code review, software architecture brainstorming:
    @xmatthias

    Beta testing and bug reporting:
    @bloodhunter4rc, Salah Lamkadem @ikonx, @ken11o2, @longyu, @paranoidandy, @smidelis, @smarm
    Juha Nykänen @suikula, Wagner Costa @wagnercosta, Johan Vlugt @Jooopieeert
    """

    def __init__(self, config: Config, live: bool=False, pair: str=''):
        if False:
            for i in range(10):
                print('nop')
        self.data: Dict[str, Any] = {}
        self.data_dictionary: Dict[str, DataFrame] = {}
        self.config = config
        self.freqai_config: Dict[str, Any] = config['freqai']
        self.full_df: DataFrame = DataFrame()
        self.append_df: DataFrame = DataFrame()
        self.data_path = Path()
        self.label_list: List = []
        self.training_features_list: List = []
        self.model_filename: str = ''
        self.backtesting_results_path = Path()
        self.backtest_predictions_folder: str = 'backtesting_predictions'
        self.live = live
        self.pair = pair
        self.keras: bool = self.freqai_config.get('keras', False)
        self.set_all_pairs()
        self.backtest_live_models = config.get('freqai_backtest_live_models', False)
        self.feature_pipeline = Pipeline()
        self.label_pipeline = Pipeline()
        self.DI_values: npt.NDArray = np.array([])
        if not self.live:
            self.full_path = self.get_full_models_path(self.config)
            if not self.backtest_live_models:
                self.full_timerange = self.create_fulltimerange(self.config['timerange'], self.freqai_config.get('train_period_days', 0))
                (self.training_timeranges, self.backtesting_timeranges) = self.split_timerange(self.full_timerange, config['freqai']['train_period_days'], config['freqai']['backtest_period_days'])
        self.data['extra_returns_per_train'] = self.freqai_config.get('extra_returns_per_train', {})
        if not self.freqai_config.get('data_kitchen_thread_count', 0):
            self.thread_count = max(int(psutil.cpu_count() * 2 - 2), 1)
        else:
            self.thread_count = self.freqai_config['data_kitchen_thread_count']
        self.train_dates: DataFrame = pd.DataFrame()
        self.unique_classes: Dict[str, list] = {}
        self.unique_class_list: list = []
        self.backtest_live_models_data: Dict[str, Any] = {}

    def set_paths(self, pair: str, trained_timestamp: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Set the paths to the data for the present coin/botloop\n        :param metadata: dict = strategy furnished pair metadata\n        :param trained_timestamp: int = timestamp of most recent training\n        '
        self.full_path = self.get_full_models_path(self.config)
        self.data_path = Path(self.full_path / f"sub-train-{pair.split('/')[0]}_{trained_timestamp}")
        return

    def make_train_test_datasets(self, filtered_dataframe: DataFrame, labels: DataFrame) -> Dict[Any, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Given the dataframe for the full history for training, split the data into\n        training and test data according to user specified parameters in configuration\n        file.\n        :param filtered_dataframe: cleaned dataframe ready to be split.\n        :param labels: cleaned labels ready to be split.\n        '
        feat_dict = self.freqai_config['feature_parameters']
        if 'shuffle' not in self.freqai_config['data_split_parameters']:
            self.freqai_config['data_split_parameters'].update({'shuffle': False})
        weights: npt.ArrayLike
        if feat_dict.get('weight_factor', 0) > 0:
            weights = self.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights = np.ones(len(filtered_dataframe))
        if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (train_features, test_features, train_labels, test_labels, train_weights, test_weights) = train_test_split(filtered_dataframe[:filtered_dataframe.shape[0]], labels, weights, **self.config['freqai']['data_split_parameters'])
        else:
            test_labels = np.zeros(2)
            test_features = pd.DataFrame()
            test_weights = np.zeros(2)
            train_features = filtered_dataframe
            train_labels = labels
            train_weights = weights
        if feat_dict['shuffle_after_split']:
            rint1 = random.randint(0, 100)
            rint2 = random.randint(0, 100)
            train_features = train_features.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_labels = train_labels.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_weights = pd.DataFrame(train_weights).sample(frac=1, random_state=rint1).reset_index(drop=True).to_numpy()[:, 0]
            test_features = test_features.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_labels = test_labels.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_weights = pd.DataFrame(test_weights).sample(frac=1, random_state=rint2).reset_index(drop=True).to_numpy()[:, 0]
        if self.freqai_config['feature_parameters'].get('reverse_train_test_order', False):
            return self.build_data_dictionary(test_features, train_features, test_labels, train_labels, test_weights, train_weights)
        else:
            return self.build_data_dictionary(train_features, test_features, train_labels, test_labels, train_weights, test_weights)

    def filter_features(self, unfiltered_df: DataFrame, training_feature_list: List, label_list: List=list(), training_filter: bool=True) -> Tuple[DataFrame, DataFrame]:
        if False:
            print('Hello World!')
        '\n        Filter the unfiltered dataframe to extract the user requested features/labels and properly\n        remove all NaNs. Any row with a NaN is removed from training dataset or replaced with\n        0s in the prediction dataset. However, prediction dataset do_predict will reflect any\n        row that had a NaN and will shield user from that prediction.\n\n        :param unfiltered_df: the full dataframe for the present training period\n        :param training_feature_list: list, the training feature list constructed by\n                                      self.build_feature_list() according to user specified\n                                      parameters in the configuration file.\n        :param labels: the labels for the dataset\n        :param training_filter: boolean which lets the function know if it is training data or\n                                prediction data to be filtered.\n        :returns:\n        :filtered_df: dataframe cleaned of NaNs and only containing the user\n        requested feature set.\n        :labels: labels cleaned of NaNs.\n        '
        filtered_df = unfiltered_df.filter(training_feature_list, axis=1)
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)
        drop_index = pd.isnull(filtered_df).any(axis=1)
        drop_index = drop_index.replace(True, 1).replace(False, 0)
        if training_filter:
            labels = unfiltered_df.filter(label_list, axis=1)
            drop_index_labels = pd.isnull(labels).any(axis=1)
            drop_index_labels = drop_index_labels.replace(True, 1).replace(False, 0)
            dates = unfiltered_df['date']
            filtered_df = filtered_df[(drop_index == 0) & (drop_index_labels == 0)]
            labels = labels[(drop_index == 0) & (drop_index_labels == 0)]
            self.train_dates = dates[(drop_index == 0) & (drop_index_labels == 0)]
            logger.info(f'{self.pair}: dropped {len(unfiltered_df) - len(filtered_df)} training points due to NaNs in populated dataset {len(unfiltered_df)}.')
            if len(unfiltered_df) == 0 and (not self.live):
                raise OperationalException(f'{self.pair}: all training data dropped due to NaNs. You likely did not download enough training data prior to your backtest timerange. Hint:\n{DOCS_LINK}/freqai-running/#downloading-data-to-cover-the-full-backtest-period')
            if 1 - len(filtered_df) / len(unfiltered_df) > 0.1 and self.live:
                worst_indicator = str(unfiltered_df.count().idxmin())
                logger.warning(f' {(1 - len(filtered_df) / len(unfiltered_df)) * 100:.0f} percent  of training data dropped due to NaNs, model may perform inconsistent with expectations. Verify {worst_indicator}')
            self.data['filter_drop_index_training'] = drop_index
        else:
            drop_index = pd.isnull(filtered_df).any(axis=1)
            self.data['filter_drop_index_prediction'] = drop_index
            filtered_df.fillna(0, inplace=True)
            drop_index = ~drop_index
            self.do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))
            if len(self.do_predict) - self.do_predict.sum() > 0:
                logger.info('dropped %s of %s prediction data points due to NaNs.', len(self.do_predict) - self.do_predict.sum(), len(filtered_df))
            labels = []
        return (filtered_df, labels)

    def build_data_dictionary(self, train_df: DataFrame, test_df: DataFrame, train_labels: DataFrame, test_labels: DataFrame, train_weights: Any, test_weights: Any) -> Dict:
        if False:
            print('Hello World!')
        self.data_dictionary = {'train_features': train_df, 'test_features': test_df, 'train_labels': train_labels, 'test_labels': test_labels, 'train_weights': train_weights, 'test_weights': test_weights, 'train_dates': self.train_dates}
        return self.data_dictionary

    def split_timerange(self, tr: str, train_split: int=28, bt_split: float=7) -> Tuple[list, list]:
        if False:
            return 10
        '\n        Function which takes a single time range (tr) and splits it\n        into sub timeranges to train and backtest on based on user input\n        tr: str, full timerange to train on\n        train_split: the period length for the each training (days). Specified in user\n        configuration file\n        bt_split: the backtesting length (days). Specified in user configuration file\n        '
        if not isinstance(train_split, int) or train_split < 1:
            raise OperationalException(f'train_period_days must be an integer greater than 0. Got {train_split}.')
        train_period_days = train_split * SECONDS_IN_DAY
        bt_period = bt_split * SECONDS_IN_DAY
        full_timerange = TimeRange.parse_timerange(tr)
        config_timerange = TimeRange.parse_timerange(self.config['timerange'])
        if config_timerange.stopts == 0:
            config_timerange.stopts = int(datetime.now(tz=timezone.utc).timestamp())
        timerange_train = copy.deepcopy(full_timerange)
        timerange_backtest = copy.deepcopy(full_timerange)
        tr_training_list = []
        tr_backtesting_list = []
        tr_training_list_timerange = []
        tr_backtesting_list_timerange = []
        first = True
        while True:
            if not first:
                timerange_train.startts = timerange_train.startts + int(bt_period)
            timerange_train.stopts = timerange_train.startts + train_period_days
            first = False
            tr_training_list.append(timerange_train.timerange_str)
            tr_training_list_timerange.append(copy.deepcopy(timerange_train))
            timerange_backtest.startts = timerange_train.stopts
            timerange_backtest.stopts = timerange_backtest.startts + int(bt_period)
            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts
            tr_backtesting_list.append(timerange_backtest.timerange_str)
            tr_backtesting_list_timerange.append(copy.deepcopy(timerange_backtest))
            if timerange_backtest.stopts == config_timerange.stopts:
                break
        return (tr_training_list_timerange, tr_backtesting_list_timerange)

    def slice_dataframe(self, timerange: TimeRange, df: DataFrame) -> DataFrame:
        if False:
            while True:
                i = 10
        '\n        Given a full dataframe, extract the user desired window\n        :param tr: timerange string that we wish to extract from df\n        :param df: Dataframe containing all candles to run the entire backtest. Here\n                   it is sliced down to just the present training period.\n        '
        if not self.live:
            df = df.loc[(df['date'] >= timerange.startdt) & (df['date'] < timerange.stopdt), :]
        else:
            df = df.loc[df['date'] >= timerange.startdt, :]
        return df

    def find_features(self, dataframe: DataFrame) -> None:
        if False:
            while True:
                i = 10
        '\n        Find features in the strategy provided dataframe\n        :param dataframe: DataFrame = strategy provided dataframe\n        :return:\n        features: list = the features to be used for training/prediction\n        '
        column_names = dataframe.columns
        features = [c for c in column_names if '%' in c]
        if not features:
            raise OperationalException('Could not find any features!')
        self.training_features_list = features

    def find_labels(self, dataframe: DataFrame) -> None:
        if False:
            return 10
        column_names = dataframe.columns
        labels = [c for c in column_names if '&' in c]
        self.label_list = labels

    def set_weights_higher_recent(self, num_weights: int) -> npt.ArrayLike:
        if False:
            print('Hello World!')
        '\n        Set weights so that recent data is more heavily weighted during\n        training than older data.\n        '
        wfactor = self.config['freqai']['feature_parameters']['weight_factor']
        weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
        return weights

    def get_predictions_to_append(self, predictions: DataFrame, do_predict: npt.ArrayLike, dataframe_backtest: DataFrame) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get backtest prediction from current backtest period\n        '
        append_df = DataFrame()
        for label in predictions.columns:
            append_df[label] = predictions[label]
            if append_df[label].dtype == object:
                continue
            if 'labels_mean' in self.data:
                append_df[f'{label}_mean'] = self.data['labels_mean'][label]
            if 'labels_std' in self.data:
                append_df[f'{label}_std'] = self.data['labels_std'][label]
        for extra_col in self.data['extra_returns_per_train']:
            append_df[f'{extra_col}'] = self.data['extra_returns_per_train'][extra_col]
        append_df['do_predict'] = do_predict
        if self.freqai_config['feature_parameters'].get('DI_threshold', 0) > 0:
            append_df['DI_values'] = self.DI_values
        dataframe_backtest.reset_index(drop=True, inplace=True)
        merged_df = pd.concat([dataframe_backtest['date'], append_df], axis=1)
        return merged_df

    def append_predictions(self, append_df: DataFrame) -> None:
        if False:
            print('Hello World!')
        '\n        Append backtest prediction from current backtest period to all previous periods\n        '
        if self.full_df.empty:
            self.full_df = append_df
        else:
            self.full_df = pd.concat([self.full_df, append_df], axis=0, ignore_index=True)

    def fill_predictions(self, dataframe):
        if False:
            print('Hello World!')
        '\n        Back fill values to before the backtesting range so that the dataframe matches size\n        when it goes back to the strategy. These rows are not included in the backtest.\n        '
        to_keep = [col for col in dataframe.columns if not col.startswith('&')]
        self.return_dataframe = pd.merge(dataframe[to_keep], self.full_df, how='left', on='date')
        self.return_dataframe[self.full_df.columns] = self.return_dataframe[self.full_df.columns].fillna(value=0)
        self.full_df = DataFrame()
        return

    def create_fulltimerange(self, backtest_tr: str, backtest_period_days: int) -> str:
        if False:
            print('Hello World!')
        if not isinstance(backtest_period_days, int):
            raise OperationalException('backtest_period_days must be an integer')
        if backtest_period_days < 0:
            raise OperationalException('backtest_period_days must be positive')
        backtest_timerange = TimeRange.parse_timerange(backtest_tr)
        if backtest_timerange.stopts == 0:
            raise OperationalException('FreqAI backtesting does not allow open ended timeranges. Please indicate the end date of your desired backtesting. timerange.')
        backtest_timerange.startts = backtest_timerange.startts - backtest_period_days * SECONDS_IN_DAY
        full_timerange = backtest_timerange.timerange_str
        config_path = Path(self.config['config_files'][0])
        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_path.resolve(), Path(self.full_path / config_path.parts[-1]))
        return full_timerange

    def check_if_model_expired(self, trained_timestamp: int) -> bool:
        if False:
            return 10
        '\n        A model age checker to determine if the model is trustworthy based on user defined\n        `expiration_hours` in the configuration file.\n        :param trained_timestamp: int = The time of training for the most recent model.\n        :return:\n            bool = If the model is expired or not.\n        '
        time = datetime.now(tz=timezone.utc).timestamp()
        elapsed_time = (time - trained_timestamp) / 3600
        max_time = self.freqai_config.get('expiration_hours', 0)
        if max_time > 0:
            return elapsed_time > max_time
        else:
            return False

    def check_if_new_training_required(self, trained_timestamp: int) -> Tuple[bool, TimeRange, TimeRange]:
        if False:
            i = 10
            return i + 15
        time = datetime.now(tz=timezone.utc).timestamp()
        trained_timerange = TimeRange()
        data_load_timerange = TimeRange()
        timeframes = self.freqai_config['feature_parameters'].get('include_timeframes')
        max_tf_seconds = 0
        for tf in timeframes:
            secs = timeframe_to_seconds(tf)
            if secs > max_tf_seconds:
                max_tf_seconds = secs
        max_period = self.config.get('startup_candle_count', 20) * 2
        additional_seconds = max_period * max_tf_seconds
        if trained_timestamp != 0:
            elapsed_time = (time - trained_timestamp) / SECONDS_IN_HOUR
            retrain = elapsed_time > self.freqai_config.get('live_retrain_hours', 0)
            if retrain:
                trained_timerange.startts = int(time - self.freqai_config.get('train_period_days', 0) * SECONDS_IN_DAY)
                trained_timerange.stopts = int(time)
                data_load_timerange.startts = int(time - self.freqai_config.get('train_period_days', 0) * SECONDS_IN_DAY - additional_seconds)
                data_load_timerange.stopts = int(time)
        else:
            trained_timerange.startts = int(time - self.freqai_config.get('train_period_days', 0) * SECONDS_IN_DAY)
            trained_timerange.stopts = int(time)
            data_load_timerange.startts = int(time - self.freqai_config.get('train_period_days', 0) * SECONDS_IN_DAY - additional_seconds)
            data_load_timerange.stopts = int(time)
            retrain = True
        return (retrain, trained_timerange, data_load_timerange)

    def set_new_model_names(self, pair: str, timestamp_id: int):
        if False:
            while True:
                i = 10
        (coin, _) = pair.split('/')
        self.data_path = Path(self.full_path / f"sub-train-{pair.split('/')[0]}_{timestamp_id}")
        self.model_filename = f'cb_{coin.lower()}_{timestamp_id}'

    def set_all_pairs(self) -> None:
        if False:
            while True:
                i = 10
        self.all_pairs = copy.deepcopy(self.freqai_config['feature_parameters'].get('include_corr_pairlist', []))
        for pair in self.config.get('exchange', '').get('pair_whitelist'):
            if pair not in self.all_pairs:
                self.all_pairs.append(pair)

    def extract_corr_pair_columns_from_populated_indicators(self, dataframe: DataFrame) -> Dict[str, DataFrame]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Find the columns of the dataframe corresponding to the corr_pairlist, save them\n        in a dictionary to be reused and attached to other pairs.\n\n        :param dataframe: fully populated dataframe (current pair + corr_pairs)\n        :return: corr_dataframes, dictionary of dataframes to be attached\n                 to other pairs in same candle.\n        '
        corr_dataframes: Dict[str, DataFrame] = {}
        pairs = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        for pair in pairs:
            pair = pair.replace(':', '')
            pair_cols = [col for col in dataframe.columns if col.startswith('%') and f'{pair}_' in col]
            if pair_cols:
                pair_cols.insert(0, 'date')
                corr_dataframes[pair] = dataframe.filter(pair_cols, axis=1)
        return corr_dataframes

    def attach_corr_pair_columns(self, dataframe: DataFrame, corr_dataframes: Dict[str, DataFrame], current_pair: str) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Attach the existing corr_pair dataframes to the current pair dataframe before training\n\n        :param dataframe: current pair strategy dataframe, indicators populated already\n        :param corr_dataframes: dictionary of saved dataframes from earlier in the same candle\n        :param current_pair: current pair to which we will attach corr pair dataframe\n        :return:\n        :dataframe: current pair dataframe of populated indicators, concatenated with corr_pairs\n                    ready for training\n        '
        pairs = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        current_pair = current_pair.replace(':', '')
        for pair in pairs:
            pair = pair.replace(':', '')
            if current_pair != pair:
                dataframe = dataframe.merge(corr_dataframes[pair], how='left', on='date')
        return dataframe

    def get_pair_data_for_features(self, pair: str, tf: str, strategy: IStrategy, corr_dataframes: dict={}, base_dataframes: dict={}, is_corr_pairs: bool=False) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        "\n        Get the data for the pair. If it's not in the dictionary, get it from the data provider\n        :param pair: str = pair to get data for\n        :param tf: str = timeframe to get data for\n        :param strategy: IStrategy = user defined strategy object\n        :param corr_dataframes: dict = dict containing the df pair dataframes\n                                (for user defined timeframes)\n        :param base_dataframes: dict = dict containing the current pair dataframes\n                                (for user defined timeframes)\n        :param is_corr_pairs: bool = whether the pair is a corr pair or not\n        :return: dataframe = dataframe containing the pair data\n        "
        if is_corr_pairs:
            dataframe = corr_dataframes[pair][tf]
            if not dataframe.empty:
                return dataframe
            else:
                dataframe = strategy.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                return dataframe
        else:
            dataframe = base_dataframes[tf]
            if not dataframe.empty:
                return dataframe
            else:
                dataframe = strategy.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                return dataframe

    def merge_features(self, df_main: DataFrame, df_to_merge: DataFrame, tf: str, timeframe_inf: str, suffix: str) -> DataFrame:
        if False:
            while True:
                i = 10
        '\n        Merge the features of the dataframe and remove HLCV and date added columns\n        :param df_main: DataFrame = main dataframe\n        :param df_to_merge: DataFrame = dataframe to merge\n        :param tf: str = timeframe of the main dataframe\n        :param timeframe_inf: str = timeframe of the dataframe to merge\n        :param suffix: str = suffix to add to the columns of the dataframe to merge\n        :return: dataframe = merged dataframe\n        '
        dataframe = merge_informative_pair(df_main, df_to_merge, tf, timeframe_inf=timeframe_inf, append_timeframe=False, suffix=suffix, ffill=True)
        skip_columns = [f'{s}_{suffix}' for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe = dataframe.drop(columns=skip_columns)
        return dataframe

    def populate_features(self, dataframe: DataFrame, pair: str, strategy: IStrategy, corr_dataframes: dict, base_dataframes: dict, is_corr_pairs: bool=False) -> DataFrame:
        if False:
            return 10
        '\n        Use the user defined strategy functions for populating features\n        :param dataframe: DataFrame = dataframe to populate\n        :param pair: str = pair to populate\n        :param strategy: IStrategy = user defined strategy object\n        :param corr_dataframes: dict = dict containing the df pair dataframes\n        :param base_dataframes: dict = dict containing the current pair dataframes\n        :param is_corr_pairs: bool = whether the pair is a corr pair or not\n        :return: dataframe = populated dataframe\n        '
        tfs: List[str] = self.freqai_config['feature_parameters'].get('include_timeframes')
        for tf in tfs:
            metadata = {'pair': pair, 'tf': tf}
            informative_df = self.get_pair_data_for_features(pair, tf, strategy, corr_dataframes, base_dataframes, is_corr_pairs)
            informative_copy = informative_df.copy()
            for t in self.freqai_config['feature_parameters']['indicator_periods_candles']:
                df_features = strategy.feature_engineering_expand_all(informative_copy.copy(), t, metadata=metadata)
                suffix = f'{t}'
                informative_df = self.merge_features(informative_df, df_features, tf, tf, suffix)
            generic_df = strategy.feature_engineering_expand_basic(informative_copy.copy(), metadata=metadata)
            suffix = 'gen'
            informative_df = self.merge_features(informative_df, generic_df, tf, tf, suffix)
            indicators = [col for col in informative_df if col.startswith('%')]
            for n in range(self.freqai_config['feature_parameters']['include_shifted_candles'] + 1):
                if n == 0:
                    continue
                df_shift = informative_df[indicators].shift(n)
                df_shift = df_shift.add_suffix('_shift-' + str(n))
                informative_df = pd.concat((informative_df, df_shift), axis=1)
            dataframe = self.merge_features(dataframe.copy(), informative_df, self.config['timeframe'], tf, f'{pair}_{tf}')
        return dataframe

    def use_strategy_to_populate_indicators(self, strategy: IStrategy, corr_dataframes: dict={}, base_dataframes: dict={}, pair: str='', prediction_dataframe: DataFrame=pd.DataFrame(), do_corr_pairs: bool=True) -> DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        Use the user defined strategy for populating indicators during retrain\n        :param strategy: IStrategy = user defined strategy object\n        :param corr_dataframes: dict = dict containing the df pair dataframes\n                                (for user defined timeframes)\n        :param base_dataframes: dict = dict containing the current pair dataframes\n                                (for user defined timeframes)\n        :param pair: str = pair to populate\n        :param prediction_dataframe: DataFrame = dataframe containing the pair data\n        used for prediction\n        :param do_corr_pairs: bool = whether to populate corr pairs or not\n        :return:\n        dataframe: DataFrame = dataframe containing populated indicators\n        '
        new_version = inspect.getsource(strategy.populate_any_indicators) == inspect.getsource(IStrategy.populate_any_indicators)
        if not new_version:
            raise OperationalException(f'You are using the `populate_any_indicators()` function which was deprecated on March 1, 2023. Please refer to the strategy migration guide to use the new feature_engineering_* methods: \n{DOCS_LINK}/strategy_migration/#freqai-strategy \nAnd the feature_engineering_* documentation: \n{DOCS_LINK}/freqai-feature-engineering/')
        tfs: List[str] = self.freqai_config['feature_parameters'].get('include_timeframes')
        pairs: List[str] = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        for tf in tfs:
            if tf not in base_dataframes:
                base_dataframes[tf] = pd.DataFrame()
            for p in pairs:
                if p not in corr_dataframes:
                    corr_dataframes[p] = {}
                if tf not in corr_dataframes[p]:
                    corr_dataframes[p][tf] = pd.DataFrame()
        if not prediction_dataframe.empty:
            dataframe = prediction_dataframe.copy()
        else:
            dataframe = base_dataframes[self.config['timeframe']].copy()
        corr_pairs: List[str] = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        dataframe = self.populate_features(dataframe.copy(), pair, strategy, corr_dataframes, base_dataframes)
        metadata = {'pair': pair}
        dataframe = strategy.feature_engineering_standard(dataframe.copy(), metadata=metadata)
        for corr_pair in corr_pairs:
            if pair == corr_pair:
                continue
            if corr_pairs and do_corr_pairs:
                dataframe = self.populate_features(dataframe.copy(), corr_pair, strategy, corr_dataframes, base_dataframes, True)
        if self.live:
            dataframe = strategy.set_freqai_targets(dataframe.copy(), metadata=metadata)
            dataframe = self.remove_special_chars_from_feature_names(dataframe)
        self.get_unique_classes_from_labels(dataframe)
        if self.config.get('reduce_df_footprint', False):
            dataframe = reduce_dataframe_footprint(dataframe)
        return dataframe

    def fit_labels(self) -> None:
        if False:
            return 10
        '\n        Fit the labels with a gaussian distribution\n        '
        import scipy as spy
        (self.data['labels_mean'], self.data['labels_std']) = ({}, {})
        for label in self.data_dictionary['train_labels'].columns:
            if self.data_dictionary['train_labels'][label].dtype == object:
                continue
            f = spy.stats.norm.fit(self.data_dictionary['train_labels'][label])
            (self.data['labels_mean'][label], self.data['labels_std'][label]) = (f[0], f[1])
        for label in self.unique_class_list:
            (self.data['labels_mean'][label], self.data['labels_std'][label]) = (0, 0)
        return

    def remove_features_from_df(self, dataframe: DataFrame) -> DataFrame:
        if False:
            return 10
        '\n        Remove the features from the dataframe before returning it to strategy. This keeps it\n        compact for Frequi purposes.\n        '
        to_keep = [col for col in dataframe.columns if not col.startswith('%') or col.startswith('%%')]
        return dataframe[to_keep]

    def get_unique_classes_from_labels(self, dataframe: DataFrame) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.find_labels(dataframe)
        for key in self.label_list:
            if dataframe[key].dtype == object:
                self.unique_classes[key] = dataframe[key].dropna().unique()
        if self.unique_classes:
            for label in self.unique_classes:
                self.unique_class_list += list(self.unique_classes[label])

    def save_backtesting_prediction(self, append_df: DataFrame) -> None:
        if False:
            while True:
                i = 10
        '\n        Save prediction dataframe from backtesting to feather file format\n        :param append_df: dataframe for backtesting period\n        '
        full_predictions_folder = Path(self.full_path / self.backtest_predictions_folder)
        if not full_predictions_folder.is_dir():
            full_predictions_folder.mkdir(parents=True, exist_ok=True)
        append_df.to_feather(self.backtesting_results_path)

    def get_backtesting_prediction(self) -> DataFrame:
        if False:
            print('Hello World!')
        '\n        Get prediction dataframe from feather file format\n        '
        append_df = pd.read_feather(self.backtesting_results_path)
        return append_df

    def check_if_backtest_prediction_is_valid(self, len_backtest_df: int) -> bool:
        if False:
            while True:
                i = 10
        '\n        Check if a backtesting prediction already exists and if the predictions\n        to append have the same size as the backtesting dataframe slice\n        :param length_backtesting_dataframe: Length of backtesting dataframe slice\n        :return:\n        :boolean: whether the prediction file is valid.\n        '
        path_to_predictionfile = Path(self.full_path / self.backtest_predictions_folder / f'{self.model_filename}_prediction.feather')
        self.backtesting_results_path = path_to_predictionfile
        file_exists = path_to_predictionfile.is_file()
        if file_exists:
            append_df = self.get_backtesting_prediction()
            if len(append_df) == len_backtest_df and 'date' in append_df:
                logger.info(f'Found backtesting prediction file at {path_to_predictionfile}')
                return True
            else:
                logger.info('A new backtesting prediction file is required. (Number of predictions is different from dataframe length or old prediction file version).')
                return False
        else:
            logger.info(f'Could not find backtesting prediction file at {path_to_predictionfile}')
            return False

    def get_full_models_path(self, config: Config) -> Path:
        if False:
            return 10
        '\n        Returns default FreqAI model path\n        :param config: Configuration dictionary\n        '
        freqai_config: Dict[str, Any] = config['freqai']
        return Path(config['user_data_dir'] / 'models' / str(freqai_config.get('identifier')))

    def remove_special_chars_from_feature_names(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            return 10
        '\n        Remove all special characters from feature strings (:)\n        :param dataframe: the dataframe that just finished indicator population. (unfiltered)\n        :return: dataframe with cleaned featrue names\n        '
        spec_chars = [':']
        for c in spec_chars:
            dataframe.columns = dataframe.columns.str.replace(c, '')
        return dataframe

    def buffer_timerange(self, timerange: TimeRange):
        if False:
            for i in range(10):
                print('nop')
        '\n        Buffer the start and end of the timerange. This is used *after* the indicators\n        are populated.\n\n        The main example use is when predicting maxima and minima, the argrelextrema\n        function  cannot know the maxima/minima at the edges of the timerange. To improve\n        model accuracy, it is best to compute argrelextrema on the full timerange\n        and then use this function to cut off the edges (buffer) by the kernel.\n\n        In another case, if the targets are set to a shifted price movement, this\n        buffer is unnecessary because the shifted candles at the end of the timerange\n        will be NaN and FreqAI will automatically cut those off of the training\n        dataset.\n        '
        buffer = self.freqai_config['feature_parameters']['buffer_train_data_candles']
        if buffer:
            timerange.stopts -= buffer * timeframe_to_seconds(self.config['timeframe'])
            timerange.startts += buffer * timeframe_to_seconds(self.config['timeframe'])
        return timerange

    def normalize_data(self, data_dictionary: Dict) -> Dict[Any, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Deprecation warning, migration assistance\n        '
        logger.warning(f'Your custom IFreqaiModel relies on the deprecated data pipeline. Please update your model to use the new data pipeline. This can be achieved by following the migration guide at {DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline We added a basic pipeline for you, but this will be removed in a future version.')
        return data_dictionary

    def denormalize_labels_from_metadata(self, df: DataFrame) -> DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        Deprecation warning, migration assistance\n        '
        logger.warning(f'Your custom IFreqaiModel relies on the deprecated data pipeline. Please update your model to use the new data pipeline. This can be achieved by following the migration guide at {DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline We added a basic pipeline for you, but this will be removed in a future version.')
        (pred_df, _, _) = self.label_pipeline.inverse_transform(df)
        return pred_df