import logging
from datetime import datetime, timedelta
from typing import Optional
from freqtrade.constants import LongShort
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn
logger = logging.getLogger(__name__)

class CooldownPeriod(IProtection):
    has_global_stop: bool = False
    has_local_stop: bool = True

    def _reason(self) -> str:
        if False:
            return 10
        '\n        LockReason to use\n        '
        return f'Cooldown period for {self.stop_duration_str}.'

    def short_desc(self) -> str:
        if False:
            print('Hello World!')
        '\n        Short method description - used for startup-messages\n        '
        return f'{self.name} - Cooldown period of {self.stop_duration_str}.'

    def _cooldown_period(self, pair: str, date_now: datetime) -> Optional[ProtectionReturn]:
        if False:
            while True:
                i = 10
        '\n        Get last trade for this pair\n        '
        look_back_until = date_now - timedelta(minutes=self._stop_duration)
        trades = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        if trades:
            trade = sorted(trades, key=lambda t: t.close_date)[-1]
            self.log_once(f'Cooldown for {pair} for {self.stop_duration_str}.', logger.info)
            until = self.calculate_lock_end([trade], self._stop_duration)
            return ProtectionReturn(lock=True, until=until, reason=self._reason())
        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Stops trading (position entering) for all pairs\n        This must evaluate to true for the whole period of the "cooldown period".\n        :return: Tuple of [bool, until, reason].\n            If true, all pairs will be locked with <reason> until <until>\n        '
        return None

    def stop_per_pair(self, pair: str, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        if False:
            while True:
                i = 10
        '\n        Stops trading (position entering) for this pair\n        This must evaluate to true for the whole period of the "cooldown period".\n        :return: Tuple of [bool, until, reason].\n            If true, this pair will be locked with <reason> until <until>\n        '
        return self._cooldown_period(pair, date_now)