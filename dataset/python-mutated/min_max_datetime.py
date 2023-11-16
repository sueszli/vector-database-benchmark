import datetime as dt
from dataclasses import InitVar, dataclass, field
from typing import Any, Mapping, Union
from airbyte_cdk.sources.declarative.datetime.datetime_parser import DatetimeParser
from airbyte_cdk.sources.declarative.interpolation.interpolated_string import InterpolatedString

@dataclass
class MinMaxDatetime:
    """
    Compares the provided date against optional minimum or maximum times. If date is earlier than
    min_date, then min_date is returned. If date is greater than max_date, then max_date is returned.
    If neither, the input date is returned.

    The timestamp format accepts the same format codes as datetime.strfptime, which are
    all the format codes required by the 1989 C standard.
    Full list of accepted format codes: https://man7.org/linux/man-pages/man3/strftime.3.html

    Attributes:
        datetime (Union[InterpolatedString, str]): InterpolatedString or string representing the datetime in the format specified by `datetime_format`
        datetime_format (str): Format of the datetime passed as argument
        min_datetime (Union[InterpolatedString, str]): Represents the minimum allowed datetime value.
        max_datetime (Union[InterpolatedString, str]): Represents the maximum allowed datetime value.
    """
    datetime: Union[InterpolatedString, str]
    parameters: InitVar[Mapping[str, Any]]
    datetime_format: str = ''
    _datetime_format: str = field(init=False, repr=False, default='')
    min_datetime: Union[InterpolatedString, str] = ''
    max_datetime: Union[InterpolatedString, str] = ''

    def __post_init__(self, parameters: Mapping[str, Any]):
        if False:
            while True:
                i = 10
        self.datetime = InterpolatedString.create(self.datetime, parameters=parameters or {})
        self._parser = DatetimeParser()
        self.min_datetime = InterpolatedString.create(self.min_datetime, parameters=parameters) if self.min_datetime else None
        self.max_datetime = InterpolatedString.create(self.max_datetime, parameters=parameters) if self.max_datetime else None

    def get_datetime(self, config, **additional_parameters) -> dt.datetime:
        if False:
            return 10
        "\n        Evaluates and returns the datetime\n        :param config: The user-provided configuration as specified by the source's spec\n        :param additional_parameters: Additional arguments to be passed to the strings for interpolation\n        :return: The evaluated datetime\n        "
        datetime_format = self._datetime_format
        if not datetime_format:
            datetime_format = '%Y-%m-%dT%H:%M:%S.%f%z'
        time = self._parser.parse(str(self.datetime.eval(config, **additional_parameters)), datetime_format)
        if self.min_datetime:
            min_time = str(self.min_datetime.eval(config, **additional_parameters))
            if min_time:
                min_time = self._parser.parse(min_time, datetime_format)
                time = max(time, min_time)
        if self.max_datetime:
            max_time = str(self.max_datetime.eval(config, **additional_parameters))
            if max_time:
                max_time = self._parser.parse(max_time, datetime_format)
                time = min(time, max_time)
        return time

    @property
    def datetime_format(self) -> str:
        if False:
            return 10
        'The format of the string representing the datetime'
        return self._datetime_format

    @datetime_format.setter
    def datetime_format(self, value: str):
        if False:
            while True:
                i = 10
        'Setter for the datetime format'
        if not isinstance(value, property):
            self._datetime_format = value