"""
This module contains the argument manager class
"""
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from typing_extensions import Self
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.exceptions import OperationalException
logger = logging.getLogger(__name__)

class TimeRange:
    """
    object defining timerange inputs.
    [start/stop]type defines if [start/stop]ts shall be used.
    if *type is None, don't use corresponding startvalue.
    """

    def __init__(self, starttype: Optional[str]=None, stoptype: Optional[str]=None, startts: int=0, stopts: int=0):
        if False:
            for i in range(10):
                print('nop')
        self.starttype: Optional[str] = starttype
        self.stoptype: Optional[str] = stoptype
        self.startts: int = startts
        self.stopts: int = stopts

    @property
    def startdt(self) -> Optional[datetime]:
        if False:
            while True:
                i = 10
        if self.startts:
            return datetime.fromtimestamp(self.startts, tz=timezone.utc)
        return None

    @property
    def stopdt(self) -> Optional[datetime]:
        if False:
            i = 10
            return i + 15
        if self.stopts:
            return datetime.fromtimestamp(self.stopts, tz=timezone.utc)
        return None

    @property
    def timerange_str(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a string representation of the timerange as used by parse_timerange.\n        Follows the format yyyymmdd-yyyymmdd - leaving out the parts that are not set.\n        '
        start = ''
        stop = ''
        if (startdt := self.startdt):
            start = startdt.strftime('%Y%m%d')
        if (stopdt := self.stopdt):
            stop = stopdt.strftime('%Y%m%d')
        return f'{start}-{stop}'

    @property
    def start_fmt(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns a string representation of the start date\n        '
        val = 'unbounded'
        if (startdt := self.startdt) is not None:
            val = startdt.strftime(DATETIME_PRINT_FORMAT)
        return val

    @property
    def stop_fmt(self) -> str:
        if False:
            return 10
        '\n        Returns a string representation of the stop date\n        '
        val = 'unbounded'
        if (stopdt := self.stopdt) is not None:
            val = stopdt.strftime(DATETIME_PRINT_FORMAT)
        return val

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Override the default Equals behavior'
        return self.starttype == other.starttype and self.stoptype == other.stoptype and (self.startts == other.startts) and (self.stopts == other.stopts)

    def subtract_start(self, seconds: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Subtracts <seconds> from startts if startts is set.\n        :param seconds: Seconds to subtract from starttime\n        :return: None (Modifies the object in place)\n        '
        if self.startts:
            self.startts = self.startts - seconds

    def adjust_start_if_necessary(self, timeframe_secs: int, startup_candles: int, min_date: datetime) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Adjust startts by <startup_candles> candles.\n        Applies only if no startup-candles have been available.\n        :param timeframe_secs: Timeframe in seconds e.g. `timeframe_to_seconds('5m')`\n        :param startup_candles: Number of candles to move start-date forward\n        :param min_date: Minimum data date loaded. Key kriterium to decide if start-time\n                         has to be moved\n        :return: None (Modifies the object in place)\n        "
        if not self.starttype or (startup_candles and min_date.timestamp() >= self.startts):
            logger.warning('Moving start-date by %s candles to account for startup time.', startup_candles)
            self.startts = int(min_date.timestamp() + timeframe_secs * startup_candles)
            self.starttype = 'date'

    @classmethod
    def parse_timerange(cls, text: Optional[str]) -> Self:
        if False:
            return 10
        '\n        Parse the value of the argument --timerange to determine what is the range desired\n        :param text: value from --timerange\n        :return: Start and End range period\n        '
        if not text:
            return cls(None, None, 0, 0)
        syntax = [('^-(\\d{8})$', (None, 'date')), ('^(\\d{8})-$', ('date', None)), ('^(\\d{8})-(\\d{8})$', ('date', 'date')), ('^-(\\d{10})$', (None, 'date')), ('^(\\d{10})-$', ('date', None)), ('^(\\d{10})-(\\d{10})$', ('date', 'date')), ('^-(\\d{13})$', (None, 'date')), ('^(\\d{13})-$', ('date', None)), ('^(\\d{13})-(\\d{13})$', ('date', 'date'))]
        for (rex, stype) in syntax:
            match = re.match(rex, text)
            if match:
                rvals = match.groups()
                index = 0
                start: int = 0
                stop: int = 0
                if stype[0]:
                    starts = rvals[index]
                    if stype[0] == 'date' and len(starts) == 8:
                        start = int(datetime.strptime(starts, '%Y%m%d').replace(tzinfo=timezone.utc).timestamp())
                    elif len(starts) == 13:
                        start = int(starts) // 1000
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == 'date' and len(stops) == 8:
                        stop = int(datetime.strptime(stops, '%Y%m%d').replace(tzinfo=timezone.utc).timestamp())
                    elif len(stops) == 13:
                        stop = int(stops) // 1000
                    else:
                        stop = int(stops)
                if start > stop > 0:
                    raise OperationalException(f'Start date is after stop date for timerange "{text}"')
                return cls(stype[0], stype[1], start, stop)
        raise OperationalException(f'Incorrect syntax for timerange "{text}"')