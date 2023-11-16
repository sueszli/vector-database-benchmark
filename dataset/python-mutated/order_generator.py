"""
This order generator is for strategies based on WeightStrategyBase
"""
from ...backtest.position import Position
from ...backtest.exchange import Exchange
import pandas as pd
import copy

class OrderGenerator:

    def generate_order_list_from_target_weight_position(self, current: Position, trade_exchange: Exchange, target_weight_position: dict, risk_degree: float, pred_start_time: pd.Timestamp, pred_end_time: pd.Timestamp, trade_start_time: pd.Timestamp, trade_end_time: pd.Timestamp) -> list:
        if False:
            return 10
        'generate_order_list_from_target_weight_position\n\n        :param current: The current position\n        :type current: Position\n        :param trade_exchange:\n        :type trade_exchange: Exchange\n        :param target_weight_position: {stock_id : weight}\n        :type target_weight_position: dict\n        :param risk_degree:\n        :type risk_degree: float\n        :param pred_start_time:\n        :type pred_start_time: pd.Timestamp\n        :param pred_end_time:\n        :type pred_end_time: pd.Timestamp\n        :param trade_start_time:\n        :type trade_start_time: pd.Timestamp\n        :param trade_end_time:\n        :type trade_end_time: pd.Timestamp\n\n        :rtype: list\n        '
        raise NotImplementedError()

class OrderGenWInteract(OrderGenerator):
    """Order Generator With Interact"""

    def generate_order_list_from_target_weight_position(self, current: Position, trade_exchange: Exchange, target_weight_position: dict, risk_degree: float, pred_start_time: pd.Timestamp, pred_end_time: pd.Timestamp, trade_start_time: pd.Timestamp, trade_end_time: pd.Timestamp) -> list:
        if False:
            i = 10
            return i + 15
        'generate_order_list_from_target_weight_position\n\n        No adjustment for for the nontradable share.\n        All the tadable value is assigned to the tadable stock according to the weight.\n        if interact == True, will use the price at trade date to generate order list\n        else, will only use the price before the trade date to generate order list\n\n        :param current:\n        :type current: Position\n        :param trade_exchange:\n        :type trade_exchange: Exchange\n        :param target_weight_position:\n        :type target_weight_position: dict\n        :param risk_degree:\n        :type risk_degree: float\n        :param pred_start_time:\n        :type pred_start_time: pd.Timestamp\n        :param pred_end_time:\n        :type pred_end_time: pd.Timestamp\n        :param trade_start_time:\n        :type trade_start_time: pd.Timestamp\n        :param trade_end_time:\n        :type trade_end_time: pd.Timestamp\n\n        :rtype: list\n        '
        if target_weight_position is None:
            return []
        current_amount_dict = current.get_stock_amount_dict()
        current_total_value = trade_exchange.calculate_amount_position_value(amount_dict=current_amount_dict, start_time=trade_start_time, end_time=trade_end_time, only_tradable=False)
        current_tradable_value = trade_exchange.calculate_amount_position_value(amount_dict=current_amount_dict, start_time=trade_start_time, end_time=trade_end_time, only_tradable=True)
        current_tradable_value += current.get_cash()
        reserved_cash = (1.0 - risk_degree) * (current_total_value + current.get_cash())
        current_tradable_value -= reserved_cash
        if current_tradable_value < 0:
            target_amount_dict = copy.deepcopy(current_amount_dict.copy())
            for stock_id in list(target_amount_dict.keys()):
                if trade_exchange.is_stock_tradable(stock_id, start_time=trade_start_time, end_time=trade_end_time):
                    del target_amount_dict[stock_id]
        else:
            current_tradable_value /= 1 + max(trade_exchange.close_cost, trade_exchange.open_cost)
            target_amount_dict = trade_exchange.generate_amount_position_from_weight_position(weight_position=target_weight_position, cash=current_tradable_value, start_time=trade_start_time, end_time=trade_end_time)
        order_list = trade_exchange.generate_order_for_target_amount_position(target_position=target_amount_dict, current_position=current_amount_dict, start_time=trade_start_time, end_time=trade_end_time)
        return order_list

class OrderGenWOInteract(OrderGenerator):
    """Order Generator Without Interact"""

    def generate_order_list_from_target_weight_position(self, current: Position, trade_exchange: Exchange, target_weight_position: dict, risk_degree: float, pred_start_time: pd.Timestamp, pred_end_time: pd.Timestamp, trade_start_time: pd.Timestamp, trade_end_time: pd.Timestamp) -> list:
        if False:
            i = 10
            return i + 15
        'generate_order_list_from_target_weight_position\n\n        generate order list directly not using the information (e.g. whether can be traded, the accurate trade price)\n         at trade date.\n        In target weight position, generating order list need to know the price of objective stock in trade date,\n        but we cannot get that\n        value when do not interact with exchange, so we check the %close price at pred_date or price recorded\n        in current position.\n\n        :param current:\n        :type current: Position\n        :param trade_exchange:\n        :type trade_exchange: Exchange\n        :param target_weight_position:\n        :type target_weight_position: dict\n        :param risk_degree:\n        :type risk_degree: float\n        :param pred_start_time:\n        :type pred_start_time: pd.Timestamp\n        :param pred_end_time:\n        :type pred_end_time: pd.Timestamp\n        :param trade_start_time:\n        :type trade_start_time: pd.Timestamp\n        :param trade_end_time:\n        :type trade_end_time: pd.Timestamp\n\n        :rtype: list of generated orders\n        '
        if target_weight_position is None:
            return []
        risk_total_value = risk_degree * current.calculate_value()
        current_stock = current.get_stock_list()
        amount_dict = {}
        for stock_id in target_weight_position:
            if trade_exchange.is_stock_tradable(stock_id=stock_id, start_time=trade_start_time, end_time=trade_end_time) and trade_exchange.is_stock_tradable(stock_id=stock_id, start_time=pred_start_time, end_time=pred_end_time):
                amount_dict[stock_id] = risk_total_value * target_weight_position[stock_id] / trade_exchange.get_close(stock_id, start_time=pred_start_time, end_time=pred_end_time)
            elif stock_id in current_stock:
                amount_dict[stock_id] = risk_total_value * target_weight_position[stock_id] / current.get_stock_price(stock_id)
            else:
                continue
        order_list = trade_exchange.generate_order_for_target_amount_position(target_position=amount_dict, current_position=current.get_stock_amount_dict(), start_time=trade_start_time, end_time=trade_end_time)
        return order_list