import os
import copy
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Text, Tuple, Union
from abc import ABC
from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.log import get_module_logger
from qlib.utils import get_pre_trading_date, load_dataset
from qlib.contrib.strategy.order_generator import OrderGenerator, OrderGenWOInteract
from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer

class BaseSignalStrategy(BaseStrategy, ABC):

    def __init__(self, *, signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame]=None, model=None, dataset=None, risk_degree: float=0.95, trade_exchange=None, level_infra=None, common_infra=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        -----------\n        signal :\n            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`\n            the decision of the strategy will base on the given signal\n        risk_degree : float\n            position percentage of total value.\n        trade_exchange : Exchange\n            exchange that provides market info, used to deal order and generate report\n            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra\n            - It allowes different trade_exchanges is used in different executions.\n            - For example:\n                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.\n                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.\n\n        '
        super().__init__(level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs)
        self.risk_degree = risk_degree
        if model is not None and dataset is not None:
            warnings.warn('`model` `dataset` is deprecated; use `signal`.', DeprecationWarning)
            signal = (model, dataset)
        self.signal: Signal = create_signal_from(signal)

    def get_risk_degree(self, trade_step=None):
        if False:
            i = 10
            return i + 15
        'get_risk_degree\n        Return the proportion of your total value you will use in investment.\n        Dynamically risk_degree will result in Market timing.\n        '
        return self.risk_degree

class TopkDropoutStrategy(BaseSignalStrategy):

    def __init__(self, *, topk, n_drop, method_sell='bottom', method_buy='top', hold_thresh=1, only_tradable=False, forbid_all_trade_at_limit=True, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        -----------\n        topk : int\n            the number of stocks in the portfolio.\n        n_drop : int\n            number of stocks to be replaced in each trading date.\n        method_sell : str\n            dropout method_sell, random/bottom.\n        method_buy : str\n            dropout method_buy, random/top.\n        hold_thresh : int\n            minimum holding days\n            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.\n        only_tradable : bool\n            will the strategy only consider the tradable stock when buying and selling.\n\n            if only_tradable:\n\n                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.\n\n            else:\n\n                strategy will make buy sell decision without checking the tradable state of the stock.\n        forbid_all_trade_at_limit : bool\n            if forbid all trades when limit_up or limit_down reached.\n\n            if forbid_all_trade_at_limit:\n\n                strategy will not do any trade when price reaches limit up/down, even not sell at limit up nor buy at\n                limit down, though allowed in reality.\n\n            else:\n\n                strategy will sell at limit up and buy ad limit down.\n        '
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit

    def generate_trade_decision(self, execute_result=None):
        if False:
            i = 10
            return i + 15
        trade_step = self.trade_calendar.get_trade_step()
        (trade_start_time, trade_end_time) = self.trade_calendar.get_step_time(trade_step)
        (pred_start_time, pred_end_time) = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)
        if self.only_tradable:

            def get_first_n(li, n, reverse=False):
                if False:
                    while True:
                        i = 10
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(stock_id=si, start_time=trade_start_time, end_time=trade_end_time):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                if False:
                    print('Hello World!')
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                if False:
                    while True:
                        i = 10
                return [si for si in li if self.trade_exchange.is_stock_tradable(stock_id=si, start_time=trade_start_time, end_time=trade_end_time)]
        else:

            def get_first_n(li, n):
                if False:
                    while True:
                        i = 10
                return list(li)[:n]

            def get_last_n(li, n):
                if False:
                    print('Hello World!')
                return list(li)[-n:]

            def filter_stock(li):
                if False:
                    print('Hello World!')
                return li
        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        if self.method_buy == 'top':
            today = get_first_n(pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index, self.n_drop + self.topk - len(last))
        elif self.method_buy == 'random':
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f'This type of input is not supported')
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        if self.method_sell == 'bottom':
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == 'random':
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell = candi
        else:
            raise NotImplementedError(f'This type of input is not supported')
        buy = today[:len(sell) + self.topk - len(last)]
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL):
                continue
            if code in sell:
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(stock_id=code, amount=sell_amount, start_time=trade_start_time, end_time=trade_end_time, direction=Order.SELL)
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    (trade_val, trade_cost, trade_price) = self.trade_exchange.deal_order(sell_order, position=current_temp)
                    cash += trade_val - trade_cost
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY):
                continue
            buy_price = self.trade_exchange.get_deal_price(stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY)
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(stock_id=code, amount=buy_amount, start_time=trade_start_time, end_time=trade_end_time, direction=Order.BUY)
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)

class WeightStrategyBase(BaseSignalStrategy):

    def __init__(self, *, order_generator_cls_or_obj=OrderGenWOInteract, **kwargs):
        if False:
            print('Hello World!')
        '\n        signal :\n            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`\n            the decision of the strategy will base on the given signal\n        trade_exchange : Exchange\n            exchange that provides market info, used to deal order and generate report\n\n            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra\n            - It allowes different trade_exchanges is used in different executions.\n            - For example:\n\n                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.\n                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.\n        '
        super().__init__(**kwargs)
        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator: OrderGenerator = order_generator_cls_or_obj()
        else:
            self.order_generator: OrderGenerator = order_generator_cls_or_obj

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if False:
            print('Hello World!')
        "\n        Generate target position from score for this date and the current position.The cash is not considered in the position\n\n        Parameters\n        -----------\n        score : pd.Series\n            pred score for this trade date, index is stock_id, contain 'score' column.\n        current : Position()\n            current position.\n        trade_start_time: pd.Timestamp\n        trade_end_time: pd.Timestamp\n        "
        raise NotImplementedError()

    def generate_trade_decision(self, execute_result=None):
        if False:
            print('Hello World!')
        trade_step = self.trade_calendar.get_trade_step()
        (trade_start_time, trade_end_time) = self.trade_calendar.get_step_time(trade_step)
        (pred_start_time, pred_end_time) = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return TradeDecisionWO([], self)
        current_temp = copy.deepcopy(self.trade_position)
        assert isinstance(current_temp, Position)
        target_weight_position = self.generate_target_weight_position(score=pred_score, current=current_temp, trade_start_time=trade_start_time, trade_end_time=trade_end_time)
        order_list = self.order_generator.generate_order_list_from_target_weight_position(current=current_temp, trade_exchange=self.trade_exchange, risk_degree=self.get_risk_degree(trade_step), target_weight_position=target_weight_position, pred_start_time=pred_start_time, pred_end_time=pred_end_time, trade_start_time=trade_start_time, trade_end_time=trade_end_time)
        return TradeDecisionWO(order_list, self)

class EnhancedIndexingStrategy(WeightStrategyBase):
    """Enhanced Indexing Strategy

    Enhanced indexing combines the arts of active management and passive management,
    with the aim of outperforming a benchmark index (e.g., S&P 500) in terms of
    portfolio return while controlling the risk exposure (a.k.a. tracking error).

    Users need to prepare their risk model data like below:

    .. code-block:: text

        ├── /path/to/riskmodel
        ├──── 20210101
        ├────── factor_exp.{csv|pkl|h5}
        ├────── factor_cov.{csv|pkl|h5}
        ├────── specific_risk.{csv|pkl|h5}
        ├────── blacklist.{csv|pkl|h5}  # optional

    The risk model data can be obtained from risk data provider. You can also use
    `qlib.model.riskmodel.structured.StructuredCovEstimator` to prepare these data.

    Args:
        riskmodel_path (str): risk model path
        name_mapping (dict): alternative file names
    """
    FACTOR_EXP_NAME = 'factor_exp.pkl'
    FACTOR_COV_NAME = 'factor_cov.pkl'
    SPECIFIC_RISK_NAME = 'specific_risk.pkl'
    BLACKLIST_NAME = 'blacklist.pkl'

    def __init__(self, *, riskmodel_root, market='csi500', turn_limit=None, name_mapping={}, optimizer_kwargs={}, verbose=False, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.logger = get_module_logger('EnhancedIndexingStrategy')
        self.riskmodel_root = riskmodel_root
        self.market = market
        self.turn_limit = turn_limit
        self.factor_exp_path = name_mapping.get('factor_exp', self.FACTOR_EXP_NAME)
        self.factor_cov_path = name_mapping.get('factor_cov', self.FACTOR_COV_NAME)
        self.specific_risk_path = name_mapping.get('specific_risk', self.SPECIFIC_RISK_NAME)
        self.blacklist_path = name_mapping.get('blacklist', self.BLACKLIST_NAME)
        self.optimizer = EnhancedIndexingOptimizer(**optimizer_kwargs)
        self.verbose = verbose
        self._riskdata_cache = {}

    def get_risk_data(self, date):
        if False:
            while True:
                i = 10
        if date in self._riskdata_cache:
            return self._riskdata_cache[date]
        root = self.riskmodel_root + '/' + date.strftime('%Y%m%d')
        if not os.path.exists(root):
            return None
        factor_exp = load_dataset(root + '/' + self.factor_exp_path, index_col=[0])
        factor_cov = load_dataset(root + '/' + self.factor_cov_path, index_col=[0])
        specific_risk = load_dataset(root + '/' + self.specific_risk_path, index_col=[0])
        if not factor_exp.index.equals(specific_risk.index):
            specific_risk = specific_risk.reindex(factor_exp.index, fill_value=specific_risk.max())
        universe = factor_exp.index.tolist()
        blacklist = []
        if os.path.exists(root + '/' + self.blacklist_path):
            blacklist = load_dataset(root + '/' + self.blacklist_path).index.tolist()
        self._riskdata_cache[date] = (factor_exp.values, factor_cov.values, specific_risk.values, universe, blacklist)
        return self._riskdata_cache[date]

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if False:
            return 10
        trade_date = trade_start_time
        pre_date = get_pre_trading_date(trade_date, future=True)
        outs = self.get_risk_data(pre_date)
        if outs is None:
            self.logger.warning(f'no risk data for {pre_date:%Y-%m-%d}, skip optimization')
            return None
        (factor_exp, factor_cov, specific_risk, universe, blacklist) = outs
        score = score.reindex(universe).fillna(score.min()).values
        cur_weight = current.get_stock_weight_dict(only_stock=False)
        cur_weight = np.array([cur_weight.get(stock, 0) for stock in universe])
        assert all(cur_weight >= 0), 'current weight has negative values'
        cur_weight = cur_weight / self.get_risk_degree(trade_date)
        if cur_weight.sum() > 1 and self.verbose:
            self.logger.warning(f'previous total holdings excess risk degree (current: {cur_weight.sum()})')
        bench_weight = D.features(D.instruments('all'), [f'${self.market}_weight'], start_time=pre_date, end_time=pre_date).squeeze()
        bench_weight.index = bench_weight.index.droplevel(level='datetime')
        bench_weight = bench_weight.reindex(universe).fillna(0).values
        tradable = D.features(D.instruments('all'), ['$volume'], start_time=pre_date, end_time=pre_date).squeeze()
        tradable.index = tradable.index.droplevel(level='datetime')
        tradable = tradable.reindex(universe).gt(0).values
        mask_force_hold = ~tradable
        mask_force_sell = np.array([stock in blacklist for stock in universe], dtype=bool)
        weight = self.optimizer(r=score, F=factor_exp, cov_b=factor_cov, var_u=specific_risk ** 2, w0=cur_weight, wb=bench_weight, mfh=mask_force_hold, mfs=mask_force_sell)
        target_weight_position = {stock: weight for (stock, weight) in zip(universe, weight) if weight > 0}
        if self.verbose:
            self.logger.info('trade date: {:%Y-%m-%d}'.format(trade_date))
            self.logger.info('number of holding stocks: {}'.format(len(target_weight_position)))
            self.logger.info('total holding weight: {:.6f}'.format(weight.sum()))
        return target_weight_position