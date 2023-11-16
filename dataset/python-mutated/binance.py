""" Binance exchange subclass """
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ccxt
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.types import OHLCVResponse, Tickers
from freqtrade.misc import deep_merge_dicts, json_load
logger = logging.getLogger(__name__)

class Binance(Exchange):
    _ft_has: Dict = {'stoploss_on_exchange': True, 'stop_price_param': 'stopPrice', 'stop_price_prop': 'stopPrice', 'stoploss_order_types': {'limit': 'stop_loss_limit'}, 'order_time_in_force': ['GTC', 'FOK', 'IOC', 'PO'], 'ohlcv_candle_limit': 1000, 'trades_pagination': 'id', 'trades_pagination_arg': 'fromId', 'l2_limit_range': [5, 10, 20, 50, 100, 500, 1000]}
    _ft_has_futures: Dict = {'stoploss_order_types': {'limit': 'stop', 'market': 'stop_market'}, 'order_time_in_force': ['GTC', 'FOK', 'IOC'], 'tickers_have_price': False, 'floor_leverage': True, 'stop_price_type_field': 'workingType', 'order_props_in_contracts': ['amount', 'cost', 'filled', 'remaining'], 'stop_price_type_value_mapping': {PriceType.LAST: 'CONTRACT_PRICE', PriceType.MARK: 'MARK_PRICE'}}
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [(TradingMode.FUTURES, MarginMode.ISOLATED)]

    def get_tickers(self, symbols: Optional[List[str]]=None, cached: bool=False) -> Tickers:
        if False:
            i = 10
            return i + 15
        tickers = super().get_tickers(symbols=symbols, cached=cached)
        if self.trading_mode == TradingMode.FUTURES:
            bidsasks = self.fetch_bids_asks(symbols, cached)
            tickers = deep_merge_dicts(bidsasks, tickers, allow_null_overrides=False)
        return tickers

    @retrier
    def additional_exchange_init(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Additional exchange initialization logic.\n        .api will be available at this point.\n        Must be overridden in child methods if required.\n        '
        try:
            if self.trading_mode == TradingMode.FUTURES and (not self._config['dry_run']):
                position_side = self._api.fapiPrivateGetPositionSideDual()
                self._log_exchange_response('position_side_setting', position_side)
                assets_margin = self._api.fapiPrivateGetMultiAssetsMargin()
                self._log_exchange_response('multi_asset_margin', assets_margin)
                msg = ''
                if position_side.get('dualSidePosition') is True:
                    msg += "\nHedge Mode is not supported by freqtrade. Please change 'Position Mode' on your binance futures account."
                if assets_margin.get('multiAssetsMargin') is True:
                    msg += "\nMulti-Asset Mode is not supported by freqtrade. Please change 'Asset Mode' on your binance futures account."
                if msg:
                    raise OperationalException(msg)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    async def _async_get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, is_new_pair: bool=False, raise_: bool=False, until_ms: Optional[int]=None) -> OHLCVResponse:
        """
        Overwrite to introduce "fast new pair" functionality by detecting the pair's listing date
        Does not work for other exchanges, which don't return the earliest data when called with "0"
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        if is_new_pair:
            x = await self._async_get_candle_history(pair, timeframe, candle_type, 0)
            if x and x[3] and x[3][0] and (x[3][0][0] > since_ms):
                since_ms = x[3][0][0]
                logger.info(f'Candle-data for {pair} available starting with {datetime.fromtimestamp(since_ms // 1000, tz=timezone.utc).isoformat()}.')
        return await super()._async_get_historic_ohlcv(pair=pair, timeframe=timeframe, since_ms=since_ms, is_new_pair=is_new_pair, raise_=raise_, candle_type=candle_type, until_ms=until_ms)

    def funding_fee_cutoff(self, open_date: datetime):
        if False:
            return 10
        '\n        Funding fees are only charged at full hours (usually every 4-8h).\n        Therefore a trade opening at 10:00:01 will not be charged a funding fee until the next hour.\n        On binance, this cutoff is 15s.\n        https://github.com/freqtrade/freqtrade/pull/5779#discussion_r740175931\n        :param open_date: The open date for a trade\n        :return: True if the date falls on a full hour, False otherwise\n        '
        return open_date.minute == 0 and open_date.second < 15

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, mm_ex_1: float=0.0, upnl_ex_1: float=0.0) -> Optional[float]:
        if False:
            while True:
                i = 10
        '\n        Important: Must be fetching data from cached values as this is used by backtesting!\n        MARGIN: https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed\n        PERPETUAL: https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93\n\n        :param pair: Pair to calculate liquidation price for\n        :param open_rate: Entry price of position\n        :param is_short: True if the trade is a short, false otherwise\n        :param amount: Absolute value of position size incl. leverage (in base currency)\n        :param stake_amount: Stake amount - Collateral in settle currency.\n        :param leverage: Leverage used for this position.\n        :param trading_mode: SPOT, MARGIN, FUTURES, etc.\n        :param margin_mode: Either ISOLATED or CROSS\n        :param wallet_balance: Amount of margin_mode in the wallet being used to trade\n            Cross-Margin Mode: crossWalletBalance\n            Isolated-Margin Mode: isolatedWalletBalance\n\n        # * Only required for Cross\n        :param mm_ex_1: (TMM)\n            Cross-Margin Mode: Maintenance Margin of all other contracts, excluding Contract 1\n            Isolated-Margin Mode: 0\n        :param upnl_ex_1: (UPNL)\n            Cross-Margin Mode: Unrealized PNL of all other contracts, excluding Contract 1.\n            Isolated-Margin Mode: 0\n        '
        side_1 = -1 if is_short else 1
        cross_vars = upnl_ex_1 - mm_ex_1 if self.margin_mode == MarginMode.CROSS else 0.0
        (mm_ratio, maintenance_amt) = self.get_maintenance_ratio_and_amt(pair, stake_amount)
        if maintenance_amt is None:
            raise OperationalException(f'Parameter maintenance_amt is required by Binance.liquidation_pricefor {self.trading_mode.value}')
        if self.trading_mode == TradingMode.FUTURES:
            return (wallet_balance + cross_vars + maintenance_amt - side_1 * amount * open_rate) / (amount * mm_ratio - side_1 * amount)
        else:
            raise OperationalException('Freqtrade only supports isolated futures for leverage trading')

    @retrier
    def load_leverage_tiers(self) -> Dict[str, List[Dict]]:
        if False:
            i = 10
            return i + 15
        if self.trading_mode == TradingMode.FUTURES:
            if self._config['dry_run']:
                leverage_tiers_path = Path(__file__).parent / 'binance_leverage_tiers.json'
                with leverage_tiers_path.open() as json_file:
                    return json_load(json_file)
            else:
                try:
                    return self._api.fetch_leverage_tiers()
                except ccxt.DDoSProtection as e:
                    raise DDosProtection(e) from e
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    raise TemporaryError(f'Could not fetch leverage amounts due to{e.__class__.__name__}. Message: {e}') from e
                except ccxt.BaseError as e:
                    raise OperationalException(e) from e
        else:
            return {}