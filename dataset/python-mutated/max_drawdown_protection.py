import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import pandas as pd
from freqtrade.constants import Config, LongShort
from freqtrade.data.metrics import calculate_max_drawdown
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn
logger = logging.getLogger(__name__)

class MaxDrawdown(IProtection):
    has_global_stop: bool = True
    has_local_stop: bool = False

    def __init__(self, config: Config, protection_config: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        super().__init__(config, protection_config)
        self._trade_limit = protection_config.get('trade_limit', 1)
        self._max_allowed_drawdown = protection_config.get('max_allowed_drawdown', 0.0)

    def short_desc(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Short method description - used for startup-messages\n        '
        return f'{self.name} - Max drawdown protection, stop trading if drawdown is > {self._max_allowed_drawdown} within {self.lookback_period_str}.'

    def _reason(self, drawdown: float) -> str:
        if False:
            i = 10
            return i + 15
        '\n        LockReason to use\n        '
        return f'{drawdown} passed {self._max_allowed_drawdown} in {self.lookback_period_str}, locking for {self.stop_duration_str}.'

    def _max_drawdown(self, date_now: datetime) -> Optional[ProtectionReturn]:
        if False:
            print('Hello World!')
        '\n        Evaluate recent trades for drawdown ...\n        '
        look_back_until = date_now - timedelta(minutes=self._lookback_period)
        trades = Trade.get_trades_proxy(is_open=False, close_date=look_back_until)
        trades_df = pd.DataFrame([trade.to_json() for trade in trades])
        if len(trades) < self._trade_limit:
            return None
        try:
            (drawdown, _, _, _, _, _) = calculate_max_drawdown(trades_df, value_col='close_profit')
        except ValueError:
            return None
        if drawdown > self._max_allowed_drawdown:
            self.log_once(f'Trading stopped due to Max Drawdown {drawdown:.2f} > {self._max_allowed_drawdown} within {self.lookback_period_str}.', logger.info)
            until = self.calculate_lock_end(trades, self._stop_duration)
            return ProtectionReturn(lock=True, until=until, reason=self._reason(drawdown))
        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        if False:
            return 10
        '\n        Stops trading (position entering) for all pairs\n        This must evaluate to true for the whole period of the "cooldown period".\n        :return: Tuple of [bool, until, reason].\n            If true, all pairs will be locked with <reason> until <until>\n        '
        return self._max_drawdown(date_now)

    def stop_per_pair(self, pair: str, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        if False:
            while True:
                i = 10
        '\n        Stops trading (position entering) for this pair\n        This must evaluate to true for the whole period of the "cooldown period".\n        :return: Tuple of [bool, until, reason].\n            If true, this pair will be locked with <reason> until <until>\n        '
        return None