import json
from datetime import datetime, timezone
import pytest
from freqtrade.persistence.trade_model import LocalTrade, Trade
from tests.conftest import create_mock_trades_usdt

@pytest.mark.usefixtures('init_persistence')
def test_trade_fromjson():
    if False:
        while True:
            i = 10
    'Test the Trade.from_json() method.'
    trade_string = '{\n        "trade_id": 25,\n        "pair": "ETH/USDT",\n        "base_currency": "ETH",\n        "quote_currency": "USDT",\n        "is_open": false,\n        "exchange": "binance",\n        "amount": 407.0,\n        "amount_requested": 102.92547026,\n        "stake_amount": 102.7494348,\n        "strategy": "SampleStrategy55",\n        "buy_tag": "Strategy2",\n        "enter_tag": "Strategy2",\n        "timeframe": 5,\n        "fee_open": 0.001,\n        "fee_open_cost": 0.1027494,\n        "fee_open_currency": "ETH",\n        "fee_close": 0.001,\n        "fee_close_cost": 0.1054944,\n        "fee_close_currency": "USDT",\n        "open_date": "2022-10-18 09:12:42",\n        "open_timestamp": 1666084362912,\n        "open_rate": 0.2518998249562391,\n        "open_rate_requested": 0.2516,\n        "open_trade_value": 102.62575199,\n        "close_date": "2022-10-18 09:45:22",\n        "close_timestamp": 1666086322208,\n        "realized_profit": 2.76315361,\n        "close_rate": 0.2592,\n        "close_rate_requested": 0.2592,\n        "close_profit": 0.026865,\n        "close_profit_pct": 2.69,\n        "close_profit_abs": 2.76315361,\n        "trade_duration_s": 1959,\n        "trade_duration": 32,\n        "profit_ratio": 0.02686,\n        "profit_pct": 2.69,\n        "profit_abs": 2.76315361,\n        "sell_reason": "no longer good",\n        "exit_reason": "no longer good",\n        "exit_order_status": "closed",\n        "stop_loss_abs": 0.1981,\n        "stop_loss_ratio": -0.216,\n        "stop_loss_pct": -21.6,\n        "stoploss_order_id": null,\n        "stoploss_last_update": "2022-10-18 09:13:42",\n        "stoploss_last_update_timestamp": 1666077222000,\n        "initial_stop_loss_abs": 0.1981,\n        "initial_stop_loss_ratio": -0.216,\n        "initial_stop_loss_pct": -21.6,\n        "min_rate": 0.2495,\n        "max_rate": 0.2592,\n        "leverage": 1.0,\n        "interest_rate": 0.0,\n        "liquidation_price": null,\n        "is_short": false,\n        "trading_mode": "spot",\n        "funding_fees": 0.0,\n        "amount_precision": 1.0,\n        "price_precision": 3.0,\n        "precision_mode": 2,\n        "contract_size": 1.0,\n        "open_order_id": null,\n        "orders": [\n            {\n                "amount": 102.0,\n                "safe_price": 0.2526,\n                "ft_order_side": "buy",\n                "order_filled_timestamp": 1666084370887,\n                "ft_is_entry": true,\n                "pair": "ETH/USDT",\n                "order_id": "78404228",\n                "status": "closed",\n                "average": 0.2526,\n                "cost": 25.7652,\n                "filled": 102.0,\n                "is_open": false,\n                "order_date": "2022-10-18 09:12:42",\n                "order_timestamp": 1666084362684,\n                "order_filled_date": "2022-10-18 09:12:50",\n                "order_type": "limit",\n                "price": 0.2526,\n                "remaining": 0.0\n            },\n            {\n                "amount": 102.0,\n                "safe_price": 0.2517,\n                "ft_order_side": "buy",\n                "order_filled_timestamp": 1666084379056,\n                "ft_is_entry": true,\n                "pair": "ETH/USDT",\n                "order_id": "78405139",\n                "status": "closed",\n                "average": 0.2517,\n                "cost": 25.6734,\n                "filled": 102.0,\n                "is_open": false,\n                "order_date": "2022-10-18 09:12:57",\n                "order_timestamp": 1666084377681,\n                "order_filled_date": "2022-10-18 09:12:59",\n                "order_type": "limit",\n                "price": 0.2517,\n                "remaining": 0.0\n            },\n            {\n                "amount": 102.0,\n                "safe_price": 0.2517,\n                "ft_order_side": "buy",\n                "order_filled_timestamp": 1666084389644,\n                "ft_is_entry": true,\n                "pair": "ETH/USDT",\n                "order_id": "78405265",\n                "status": "closed",\n                "average": 0.2517,\n                "cost": 25.6734,\n                "filled": 102.0,\n                "is_open": false,\n                "order_date": "2022-10-18 09:13:03",\n                "order_timestamp": 1666084383295,\n                "order_filled_date": "2022-10-18 09:13:09",\n                "order_type": "limit",\n                "price": 0.2517,\n                "remaining": 0.0\n            },\n            {\n                "amount": 102.0,\n                "safe_price": 0.2516,\n                "ft_order_side": "buy",\n                "order_filled_timestamp": 1666084723521,\n                "ft_is_entry": true,\n                "pair": "ETH/USDT",\n                "order_id": "78405395",\n                "status": "closed",\n                "average": 0.2516,\n                "cost": 25.6632,\n                "filled": 102.0,\n                "is_open": false,\n                "order_date": "2022-10-18 09:13:13",\n                "order_timestamp": 1666084393920,\n                "order_filled_date": "2022-10-18 09:18:43",\n                "order_type": "limit",\n                "price": 0.2516,\n                "remaining": 0.0\n            },\n            {\n                "amount": 407.0,\n                "safe_price": 0.2592,\n                "ft_order_side": "sell",\n                "order_filled_timestamp": 1666086322198,\n                "ft_is_entry": false,\n                "pair": "ETH/USDT",\n                "order_id": "78432649",\n                "status": "closed",\n                "average": 0.2592,\n                "cost": 105.4944,\n                "filled": 407.0,\n                "is_open": false,\n                "order_date": "2022-10-18 09:45:21",\n                "order_timestamp": 1666086321435,\n                "order_filled_date": "2022-10-18 09:45:22",\n                "order_type": "market",\n                "price": 0.2592,\n                "remaining": 0.0,\n                "funding_fee": -0.055\n            }\n        ]\n    }'
    trade = Trade.from_json(trade_string)
    Trade.session.add(trade)
    Trade.commit()
    assert trade.id == 25
    assert trade.pair == 'ETH/USDT'
    assert trade.open_date_utc == datetime(2022, 10, 18, 9, 12, 42, tzinfo=timezone.utc)
    assert isinstance(trade.open_date, datetime)
    assert trade.exit_reason == 'no longer good'
    assert trade.realized_profit == 2.76315361
    assert trade.precision_mode == 2
    assert trade.amount_precision == 1.0
    assert trade.contract_size == 1.0
    assert len(trade.orders) == 5
    last_o = trade.orders[-1]
    assert last_o.order_filled_utc == datetime(2022, 10, 18, 9, 45, 22, tzinfo=timezone.utc)
    assert isinstance(last_o.order_date, datetime)
    assert last_o.funding_fee == -0.055

@pytest.mark.usefixtures('init_persistence')
def test_trade_serialize_load_back(fee):
    if False:
        for i in range(10):
            print('nop')
    create_mock_trades_usdt(fee, None)
    t = Trade.get_trades([Trade.id == 1]).first()
    assert t.id == 1
    t.funding_fees = 0.025
    t.orders[0].funding_fee = 0.0125
    assert len(t.orders) == 2
    Trade.commit()
    tjson = t.to_json(False)
    assert isinstance(tjson, dict)
    trade_string = json.dumps(tjson)
    trade = Trade.from_json(trade_string)
    assert trade.id == t.id
    assert trade.funding_fees == t.funding_fees
    assert len(trade.orders) == len(t.orders)
    assert trade.orders[0].funding_fee == t.orders[0].funding_fee
    excluded = ['trade_id', 'quote_currency', 'open_timestamp', 'close_timestamp', 'realized_profit_ratio', 'close_profit_pct', 'trade_duration_s', 'trade_duration', 'profit_ratio', 'profit_pct', 'profit_abs', 'stop_loss_abs', 'initial_stop_loss_abs', 'orders']
    failed = []
    for (obj, value) in tjson.items():
        if obj in excluded:
            continue
        tattr = getattr(trade, obj, None)
        if isinstance(tattr, datetime):
            tattr = tattr.strftime('%Y-%m-%d %H:%M:%S')
        if tattr != value:
            failed.append((obj, tattr, value))
    assert tjson.get('trade_id') == trade.id
    assert tjson.get('quote_currency') == trade.stake_currency
    assert tjson.get('stop_loss_abs') == trade.stop_loss
    assert tjson.get('initial_stop_loss_abs') == trade.initial_stop_loss
    excluded_o = ['order_filled_timestamp', 'ft_is_entry', 'pair', 'is_open', 'order_timestamp']
    order_obj = trade.orders[0]
    for (obj, value) in tjson['orders'][0].items():
        if obj in excluded_o:
            continue
        tattr = getattr(order_obj, obj, None)
        if isinstance(tattr, datetime):
            tattr = tattr.strftime('%Y-%m-%d %H:%M:%S')
        if tattr != value:
            failed.append((obj, tattr, value))
    assert tjson['orders'][0]['pair'] == order_obj.ft_pair
    assert not failed
    trade2 = LocalTrade.from_json(trade_string)
    assert len(trade2.orders) == len(t.orders)
    trade3 = LocalTrade.from_json(trade_string)
    assert len(trade3.orders) == len(t.orders)