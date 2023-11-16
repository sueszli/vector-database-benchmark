import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pandas import DataFrame
from freqtrade.configuration import TimeRange
logger = logging.getLogger(__name__)

class VarHolder:
    timerange: TimeRange
    data: DataFrame
    indicators: Dict[str, DataFrame]
    result: DataFrame
    compared: DataFrame
    from_dt: datetime
    to_dt: datetime
    compared_dt: datetime
    timeframe: str
    startup_candle: int

class BaseAnalysis:

    def __init__(self, config: Dict[str, Any], strategy_obj: Dict):
        if False:
            while True:
                i = 10
        self.failed_bias_check = True
        self.full_varHolder = VarHolder()
        self.exchange: Optional[Any] = None
        self._fee = None
        self.local_config = deepcopy(config)
        self.local_config['strategy'] = strategy_obj['name']
        self.strategy_obj = strategy_obj

    @staticmethod
    def dt_to_timestamp(dt: datetime):
        if False:
            print('Hello World!')
        timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
        return timestamp

    def fill_full_varholder(self):
        if False:
            i = 10
            return i + 15
        self.full_varHolder = VarHolder()
        parsed_timerange = TimeRange.parse_timerange(self.local_config['timerange'])
        if parsed_timerange.startdt is None:
            self.full_varHolder.from_dt = datetime.fromtimestamp(0, tz=timezone.utc)
        else:
            self.full_varHolder.from_dt = parsed_timerange.startdt
        if parsed_timerange.stopdt is None:
            self.full_varHolder.to_dt = datetime.utcnow()
        else:
            self.full_varHolder.to_dt = parsed_timerange.stopdt
        self.prepare_data(self.full_varHolder, self.local_config['pairs'])

    def start(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.fill_full_varholder()