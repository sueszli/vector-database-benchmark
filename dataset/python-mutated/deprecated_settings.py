"""
Functions to handle deprecated settings
"""
import logging
from typing import Optional
from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
logger = logging.getLogger(__name__)

def check_conflicting_settings(config: Config, section_old: Optional[str], name_old: str, section_new: Optional[str], name_new: str) -> None:
    if False:
        while True:
            i = 10
    section_new_config = config.get(section_new, {}) if section_new else config
    section_old_config = config.get(section_old, {}) if section_old else config
    if name_new in section_new_config and name_old in section_old_config:
        new_name = f'{section_new}.{name_new}' if section_new else f'{name_new}'
        old_name = f'{section_old}.{name_old}' if section_old else f'{name_old}'
        raise OperationalException(f'Conflicting settings `{new_name}` and `{old_name}` (DEPRECATED) detected in the configuration file. This deprecated setting will be removed in the next versions of Freqtrade. Please delete it from your configuration and use the `{new_name}` setting instead.')

def process_removed_setting(config: Config, section1: str, name1: str, section2: Optional[str], name2: str) -> None:
    if False:
        while True:
            i = 10
    '\n    :param section1: Removed section\n    :param name1: Removed setting name\n    :param section2: new section for this key\n    :param name2: new setting name\n    '
    section1_config = config.get(section1, {})
    if name1 in section1_config:
        section_2 = f'{section2}.{name2}' if section2 else f'{name2}'
        raise OperationalException(f'Setting `{section1}.{name1}` has been moved to `{section_2}. Please delete it from your configuration and use the `{section_2}` setting instead.')

def process_deprecated_setting(config: Config, section_old: Optional[str], name_old: str, section_new: Optional[str], name_new: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    check_conflicting_settings(config, section_old, name_old, section_new, name_new)
    section_old_config = config.get(section_old, {}) if section_old else config
    if name_old in section_old_config:
        section_1 = f'{section_old}.{name_old}' if section_old else f'{name_old}'
        section_2 = f'{section_new}.{name_new}' if section_new else f'{name_new}'
        logger.warning(f'DEPRECATED: The `{section_1}` setting is deprecated and will be removed in the next versions of Freqtrade. Please use the `{section_2}` setting in your configuration instead.')
        section_new_config = config.get(section_new, {}) if section_new else config
        section_new_config[name_new] = section_old_config[name_old]
        del section_old_config[name_old]

def process_temporary_deprecated_settings(config: Config) -> None:
    if False:
        while True:
            i = 10
    process_deprecated_setting(config, 'ask_strategy', 'ignore_buying_expired_candle_after', None, 'ignore_buying_expired_candle_after')
    process_deprecated_setting(config, None, 'forcebuy_enable', None, 'force_entry_enable')
    if config.get('telegram'):
        process_deprecated_setting(config['telegram'], 'notification_settings', 'sell', 'notification_settings', 'exit')
        process_deprecated_setting(config['telegram'], 'notification_settings', 'sell_fill', 'notification_settings', 'exit_fill')
        process_deprecated_setting(config['telegram'], 'notification_settings', 'sell_cancel', 'notification_settings', 'exit_cancel')
        process_deprecated_setting(config['telegram'], 'notification_settings', 'buy', 'notification_settings', 'entry')
        process_deprecated_setting(config['telegram'], 'notification_settings', 'buy_fill', 'notification_settings', 'entry_fill')
        process_deprecated_setting(config['telegram'], 'notification_settings', 'buy_cancel', 'notification_settings', 'entry_cancel')
    if config.get('webhook'):
        process_deprecated_setting(config, 'webhook', 'webhookbuy', 'webhook', 'webhookentry')
        process_deprecated_setting(config, 'webhook', 'webhookbuycancel', 'webhook', 'webhookentrycancel')
        process_deprecated_setting(config, 'webhook', 'webhookbuyfill', 'webhook', 'webhookentryfill')
        process_deprecated_setting(config, 'webhook', 'webhooksell', 'webhook', 'webhookexit')
        process_deprecated_setting(config, 'webhook', 'webhooksellcancel', 'webhook', 'webhookexitcancel')
        process_deprecated_setting(config, 'webhook', 'webhooksellfill', 'webhook', 'webhookexitfill')
    process_removed_setting(config, 'experimental', 'use_sell_signal', None, 'use_exit_signal')
    process_removed_setting(config, 'experimental', 'sell_profit_only', None, 'exit_profit_only')
    process_removed_setting(config, 'experimental', 'ignore_roi_if_buy_signal', None, 'ignore_roi_if_entry_signal')
    process_removed_setting(config, 'ask_strategy', 'use_sell_signal', None, 'use_exit_signal')
    process_removed_setting(config, 'ask_strategy', 'sell_profit_only', None, 'exit_profit_only')
    process_removed_setting(config, 'ask_strategy', 'sell_profit_offset', None, 'exit_profit_offset')
    process_removed_setting(config, 'ask_strategy', 'ignore_roi_if_buy_signal', None, 'ignore_roi_if_entry_signal')
    if config.get('edge', {}).get('enabled', False) and 'capital_available_percentage' in config.get('edge', {}):
        raise OperationalException("DEPRECATED: Using 'edge.capital_available_percentage' has been deprecated in favor of 'tradable_balance_ratio'. Please migrate your configuration to 'tradable_balance_ratio' and remove 'capital_available_percentage' from the edge configuration.")
    if 'ticker_interval' in config:
        raise OperationalException("DEPRECATED: 'ticker_interval' detected. Please use 'timeframe' instead of 'ticker_interval.")
    if 'protections' in config:
        logger.warning("DEPRECATED: Setting 'protections' in the configuration is deprecated.")