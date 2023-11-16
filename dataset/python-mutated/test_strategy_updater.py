import re
import shutil
import sys
from pathlib import Path
import pytest
from freqtrade.commands.strategy_utils_commands import start_strategy_update
from freqtrade.strategy.strategyupdater import StrategyUpdater
from tests.conftest import get_args
if sys.version_info < (3, 9):
    pytest.skip('StrategyUpdater is not compatible with Python 3.8', allow_module_level=True)

def test_strategy_updater_start(user_dir, capsys) -> None:
    if False:
        for i in range(10):
            print('nop')
    teststrats = Path(__file__).parent / 'strategy/strats'
    tmpdirp = Path(user_dir) / 'strategies'
    tmpdirp.mkdir(parents=True, exist_ok=True)
    shutil.copy(teststrats / 'strategy_test_v2.py', tmpdirp)
    old_code = (teststrats / 'strategy_test_v2.py').read_text()
    args = ['strategy-updater', '--userdir', str(user_dir), '--strategy-list', 'StrategyTestV2']
    pargs = get_args(args)
    pargs['config'] = None
    start_strategy_update(pargs)
    assert Path(user_dir / 'strategies_orig_updater').exists()
    assert Path(user_dir / 'strategies_orig_updater' / 'strategy_test_v2.py').exists()
    new_file = tmpdirp / 'strategy_test_v2.py'
    assert new_file.exists()
    new_code = new_file.read_text()
    assert 'INTERFACE_VERSION = 3' in new_code
    assert 'INTERFACE_VERSION = 2' in old_code
    captured = capsys.readouterr()
    assert 'Conversion of strategy_test_v2.py started.' in captured.out
    assert re.search('Conversion of strategy_test_v2\\.py took .* seconds', captured.out)

def test_strategy_updater_methods(default_conf, caplog) -> None:
    if False:
        return 10
    instance_strategy_updater = StrategyUpdater()
    modified_code1 = instance_strategy_updater.update_code('\nclass testClass(IStrategy):\n    def populate_buy_trend():\n        pass\n    def populate_sell_trend():\n        pass\n    def check_buy_timeout():\n        pass\n    def check_sell_timeout():\n        pass\n    def custom_sell():\n        pass\n')
    assert 'populate_entry_trend' in modified_code1
    assert 'populate_exit_trend' in modified_code1
    assert 'check_entry_timeout' in modified_code1
    assert 'check_exit_timeout' in modified_code1
    assert 'custom_exit' in modified_code1
    assert 'INTERFACE_VERSION = 3' in modified_code1

def test_strategy_updater_params(default_conf, caplog) -> None:
    if False:
        i = 10
        return i + 15
    instance_strategy_updater = StrategyUpdater()
    modified_code2 = instance_strategy_updater.update_code("\nticker_interval = '15m'\nbuy_some_parameter = IntParameter(space='buy')\nsell_some_parameter = IntParameter(space='sell')\n")
    assert 'timeframe' in modified_code2
    assert "space='buy'" in modified_code2
    assert "space='sell'" in modified_code2

def test_strategy_updater_constants(default_conf, caplog) -> None:
    if False:
        print('Hello World!')
    instance_strategy_updater = StrategyUpdater()
    modified_code3 = instance_strategy_updater.update_code('\nuse_sell_signal = True\nsell_profit_only = True\nsell_profit_offset = True\nignore_roi_if_buy_signal = True\nforcebuy_enable = True\n')
    assert 'use_exit_signal' in modified_code3
    assert 'exit_profit_only' in modified_code3
    assert 'exit_profit_offset' in modified_code3
    assert 'ignore_roi_if_entry_signal' in modified_code3
    assert 'force_entry_enable' in modified_code3

def test_strategy_updater_df_columns(default_conf, caplog) -> None:
    if False:
        for i in range(10):
            print('nop')
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code('\ndataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_1")\ndataframe.loc[reduce(lambda x, y: x & y, conditions), \'sell\'] = 1\n')
    assert 'enter_long' in modified_code
    assert 'exit_long' in modified_code
    assert 'enter_tag' in modified_code

def test_strategy_updater_method_params(default_conf, caplog) -> None:
    if False:
        return 10
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code('\ndef confirm_trade_exit(sell_reason: str):\n    nr_orders = trade.nr_of_successful_buys\n    pass\n    ')
    assert 'exit_reason' in modified_code
    assert 'nr_orders = trade.nr_of_successful_entries' in modified_code

def test_strategy_updater_dicts(default_conf, caplog) -> None:
    if False:
        return 10
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code("\norder_time_in_force = {\n    'buy': 'gtc',\n    'sell': 'ioc'\n}\norder_types = {\n    'buy': 'limit',\n    'sell': 'market',\n    'stoploss': 'market',\n    'stoploss_on_exchange': False\n}\nunfilledtimeout = {\n    'buy': 1,\n    'sell': 2\n}\n")
    assert "'entry': 'gtc'" in modified_code
    assert "'exit': 'ioc'" in modified_code
    assert "'entry': 'limit'" in modified_code
    assert "'exit': 'market'" in modified_code
    assert "'entry': 1" in modified_code
    assert "'exit': 2" in modified_code

def test_strategy_updater_comparisons(default_conf, caplog) -> None:
    if False:
        for i in range(10):
            print('nop')
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code("\ndef confirm_trade_exit(sell_reason):\n    if (sell_reason == 'stop_loss'):\n        pass\n")
    assert 'exit_reason' in modified_code
    assert "exit_reason == 'stop_loss'" in modified_code

def test_strategy_updater_strings(default_conf, caplog) -> None:
    if False:
        return 10
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code("\nsell_reason == 'sell_signal'\nsell_reason == 'force_sell'\nsell_reason == 'emergency_sell'\n")
    assert 'exit_signal' in modified_code
    assert 'exit_reason' in modified_code
    assert 'force_exit' in modified_code
    assert 'emergency_exit' in modified_code

def test_strategy_updater_comments(default_conf, caplog) -> None:
    if False:
        return 10
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code('\n# This is the 1st comment\nimport talib.abstract as ta\n# This is the 2nd comment\nimport freqtrade.vendor.qtpylib.indicators as qtpylib\n\n\nclass someStrategy(IStrategy):\n    INTERFACE_VERSION = 2\n    # This is the 3rd comment\n    # This attribute will be overridden if the config file contains "minimal_roi"\n    minimal_roi = {\n        "0": 0.50\n    }\n\n    # This is the 4th comment\n    stoploss = -0.1\n')
    assert 'This is the 1st comment' in modified_code
    assert 'This is the 2nd comment' in modified_code
    assert 'This is the 3rd comment' in modified_code
    assert 'INTERFACE_VERSION = 3' in modified_code