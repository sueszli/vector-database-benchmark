import sys
import inspect
import datetime
import six
import pandas as pd
from typing import Iterable
from functools import wraps
from collections import OrderedDict
from contextlib import contextmanager
from dateutil.parser import parse as parse_date
from rqalpha.utils.exception import RQInvalidArgument, RQTypeError, RQApiNotSupportedError
from rqalpha.model.instrument import Instrument
from rqalpha.environment import Environment
from rqalpha.const import INSTRUMENT_TYPE, EXC_TYPE
from rqalpha.utils import unwrapper, INST_TYPE_IN_STOCK_ACCOUNT
from rqalpha.utils.i18n import gettext as _
from rqalpha.utils.exception import patch_system_exc, EXC_EXT_NAME
from rqalpha.utils.logger import user_system_log
main_contract_warning_flag = True
index_contract_warning_flag = True

class ArgumentChecker(object):

    def __init__(self, arg_name, pre_check):
        if False:
            print('Hello World!')
        self._arg_name = arg_name
        self._pre_check = pre_check
        self._rules = []

    def is_instance_of(self, types):
        if False:
            for i in range(10):
                print('nop')

        def check_is_instance_of(func_name, value):
            if False:
                print('Hello World!')
            if not isinstance(value, types):
                raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a value of type {}, got {} (type: {})').format(func_name, self._arg_name, types, value, type(value)))
        self._rules.append(check_is_instance_of)
        return self

    def raise_invalid_instrument_error(self, func_name, value):
        if False:
            while True:
                i = 10
        return self.raise_instrument_error(func_name, value, _('valid order_book_id/instrument'))

    def raise_not_valid_stock_error(self, func_name, value):
        if False:
            print('Hello World!')
        return self.raise_instrument_error(func_name, value, _('valid stock order_book_id/instrument'))

    def raise_not_valid_future_error(self, func_name, value):
        if False:
            return 10
        return self.raise_instrument_error(func_name, value, _('valid future order_book_id/instrument'))

    def raise_instrument_not_listed_error(self, func_name, value):
        if False:
            while True:
                i = 10
        return self.raise_instrument_error(func_name, value, _('listed order_book_id/instrument'))

    def raise_instrument_error(self, func_name, value, instrument_info):
        if False:
            while True:
                i = 10
        raise RQInvalidArgument(_(u'function {}: invalid {} argument, expected a {}, got {} (type: {})').format(func_name, self._arg_name, instrument_info, value, type(value)))

    def _is_valid_instrument(self, func_name, value):
        if False:
            return 10
        instrument = None
        if isinstance(value, six.string_types):
            instrument = Environment.get_instance().get_instrument(value)
        elif isinstance(value, Instrument):
            instrument = value
        if instrument is None:
            self.raise_invalid_instrument_error(func_name, value)
        return instrument

    def is_valid_instrument(self, valid_instrument_types=None):
        if False:
            i = 10
            return i + 15

        def check_is_valid_instrument(func_name, value):
            if False:
                return 10
            instrument = None
            if isinstance(value, six.string_types):
                instrument = Environment.get_instance().get_instrument(value)
            elif isinstance(value, Instrument):
                instrument = value
            if instrument is None:
                self.raise_invalid_instrument_error(func_name, value)
            if valid_instrument_types and instrument.type not in valid_instrument_types:
                raise RQInvalidArgument(_(u'function {}: invalid {} argument, expected instrument with types {}, got instrument with type {}').format(func_name, self._arg_name, valid_instrument_types, instrument.type))
            return instrument
        self._rules.append(check_is_valid_instrument)
        return self

    def _is_listed_instrument(self, func_name, value):
        if False:
            i = 10
            return i + 15
        instrument = self._is_valid_instrument(func_name, value)
        if not instrument.listed:
            self.raise_instrument_not_listed_error(func_name, value)

    def is_listed_instrument(self):
        if False:
            i = 10
            return i + 15
        self._rules.append(self._is_listed_instrument)
        return self

    def _is_valid_stock(self, func_name, value):
        if False:
            for i in range(10):
                print('nop')
        instrument = self._is_valid_instrument(func_name, value)
        if instrument.type not in INST_TYPE_IN_STOCK_ACCOUNT:
            self.raise_not_valid_stock_error(func_name, value)

    def is_valid_stock(self):
        if False:
            print('Hello World!')
        self._rules.append(self._is_valid_stock)
        return self

    def _is_valid_future(self, func_name, value):
        if False:
            return 10
        instrument = self._is_valid_instrument(func_name, value)
        if instrument.type != INSTRUMENT_TYPE.FUTURE:
            self.raise_not_valid_future_error(func_name, value)

    def is_valid_future(self):
        if False:
            while True:
                i = 10
        self._rules.append(self._is_valid_future)
        return self

    def _is_number(self, func_name, value):
        if False:
            for i in range(10):
                print('nop')
        try:
            v = float(value)
        except (ValueError, TypeError):
            raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a number, got {} (type: {})').format(func_name, self._arg_name, value, type(value)))

    def is_number(self):
        if False:
            return 10
        self._rules.append(self._is_number)
        return self

    def deprecated(self, hint='deprecated'):
        if False:
            for i in range(10):
                print('nop')

        def inner(func_name, value):
            if False:
                while True:
                    i = 10
            if value is not None:
                content = '{} param {} is deprecated. {}'.format(func_name, self._arg_name, hint)
                user_system_log.warning(content)
        self._rules.append(inner)
        return self

    def is_in(self, valid_values, ignore_none=True):
        if False:
            return 10

        def check_is_in(func_name, value):
            if False:
                i = 10
                return i + 15
            if ignore_none and value is None:
                return
            if value not in valid_values:
                raise RQInvalidArgument(_(u'function {}: invalid {} argument, valid: {}, got {} (type: {})').format(func_name, self._arg_name, repr(valid_values), value, type(value)))
        self._rules.append(check_is_in)
        return self

    def are_valid_fields(self, valid_fields, ignore_none=True):
        if False:
            for i in range(10):
                print('nop')
        valid_fields = set(valid_fields)

        def check_are_valid_fields(func_name, fields):
            if False:
                return 10
            if isinstance(fields, six.string_types):
                if fields not in valid_fields:
                    raise RQInvalidArgument(_(u'function {}: invalid {} argument, valid fields are {}, got {} (type: {})').format(func_name, self._arg_name, repr(valid_fields), fields, type(fields)))
                return
            if fields is None and ignore_none:
                return
            if isinstance(fields, list):
                invalid_fields = [field for field in fields if field not in valid_fields]
                if invalid_fields:
                    raise RQInvalidArgument(_(u'function {}: invalid field {}, valid fields are {}, got {} (type: {})').format(func_name, invalid_fields, repr(valid_fields), fields, type(fields)))
                return
            raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a string or a list of string, got {} (type: {})').format(func_name, self._arg_name, repr(fields), type(fields)))
        self._rules.append(check_are_valid_fields)
        return self

    def _are_valid_instruments(self, func_name, values):
        if False:
            i = 10
            return i + 15
        if isinstance(values, (six.string_types, Instrument)):
            self._is_valid_instrument(func_name, values)
        elif isinstance(values, list):
            for v in values:
                self._is_valid_instrument(func_name, v)
        else:
            raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a string or a list of string, got {} (type: {})').format(func_name, self._arg_name, repr(values), type(values)))

    def are_valid_instruments(self, ignore_none=False):
        if False:
            i = 10
            return i + 15

        def check_are_valid_instruments(func_name, values):
            if False:
                return 10
            if values is None and ignore_none:
                return
            return self._are_valid_instruments(func_name, values)
        self._rules.append(check_are_valid_instruments)
        return self

    def is_valid_date(self, ignore_none=True):
        if False:
            while True:
                i = 10

        def check_is_valid_date(func_name, value):
            if False:
                return 10
            if ignore_none and value is None:
                return None
            if isinstance(value, (datetime.date, pd.Timestamp)):
                return
            if isinstance(value, six.string_types):
                try:
                    v = parse_date(value)
                    return
                except ValueError:
                    raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a valid date, got {} (type: {})').format(func_name, self._arg_name, value, type(value)))
            raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a valid date, got {} (type: {})').format(func_name, self._arg_name, value, type(value)))
        self._rules.append(check_is_valid_date)
        return self

    def is_greater_or_equal_than(self, low):
        if False:
            return 10

        def check_greater_or_equal_than(func_name, value):
            if False:
                while True:
                    i = 10
            if isinstance(value, (int, float)) and value < low:
                raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a value >= {}, got {} (type: {})').format(func_name, self._arg_name, low, value, type(value)))
        self._rules.append(check_greater_or_equal_than)
        return self

    def is_greater_than(self, low):
        if False:
            while True:
                i = 10

        def check_greater_than(func_name, value):
            if False:
                while True:
                    i = 10
            if isinstance(value, (int, float)) and value <= low:
                raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a value > {}, got {} (type: {})').format(func_name, self._arg_name, low, value, type(value)))
        self._rules.append(check_greater_than)
        return self

    def is_less_or_equal_than(self, high):
        if False:
            print('Hello World!')

        def check_less_or_equal_than(func_name, value):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(value, (int, float)) and value > high:
                raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a value <= {}, got {} (type: {})').format(func_name, self._arg_name, high, value, type(value)))
        self._rules.append(check_less_or_equal_than)
        return self

    def is_less_than(self, high):
        if False:
            for i in range(10):
                print('nop')

        def check_less_than(func_name, value):
            if False:
                i = 10
                return i + 15
            if isinstance(value, (int, float)) and value >= high:
                raise RQInvalidArgument(_(u'function {}: invalid {} argument, expect a value < {}, got {} (type: {})').format(func_name, self._arg_name, high, value, type(value)))
        self._rules.append(check_less_than)
        return self

    def _is_valid_interval(self, func_name, value):
        if False:
            for i in range(10):
                print('nop')
        valid = isinstance(value, six.string_types) and value[-1] in {'d', 'm', 'q', 'y'}
        if valid:
            try:
                valid = int(value[:-1]) > 0
            except (ValueError, TypeError):
                valid = False
        if not valid:
            raise RQInvalidArgument(_(u"function {}: invalid {} argument, interval should be in form of '1d', '3m', '4q', '2y', got {} (type: {})").format(func_name, self.arg_name, value, type(value)))

    def is_valid_interval(self):
        if False:
            print('Hello World!')
        self._rules.append(self._is_valid_interval)
        return self

    def _is_valid_quarter(self, func_name, value):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            valid = True
        else:
            valid = isinstance(value, six.string_types) and value[-2] == 'q'
            if valid:
                try:
                    valid = 1990 <= int(value[:-2]) <= 2099 and 1 <= int(value[-1]) <= 4
                except (ValueError, TypeError):
                    valid = False
        if not valid:
            raise RQInvalidArgument(_(u"function {}: invalid {} argument, quarter should be in form of '2012q3', got {} (type: {})").format(func_name, self.arg_name, value, type(value)))

    def is_valid_quarter(self):
        if False:
            print('Hello World!')
        self._rules.append(self._is_valid_quarter)
        return self

    def _are_valid_query_entities(self, func_name, entities):
        if False:
            while True:
                i = 10
        from sqlalchemy.orm.attributes import InstrumentedAttribute
        for e in entities:
            if not isinstance(e, InstrumentedAttribute):
                raise RQInvalidArgument(_(u'function {}: invalid {} argument, should be entity like Fundamentals.balance_sheet.total_equity, got {} (type: {})').format(func_name, self.arg_name, e, type(e)))

    def are_valid_query_entities(self):
        if False:
            while True:
                i = 10
        self._rules.append(self._are_valid_query_entities)
        return self

    def _is_valid_frequency(self, func_name, value):
        if False:
            while True:
                i = 10
        valid = isinstance(value, six.string_types) and value[-1] in ('d', 'm', 'w')
        if valid:
            try:
                valid = int(value[:-1]) > 0
            except (ValueError, TypeError):
                valid = False
        if not valid:
            raise RQInvalidArgument(_(u"function {}: invalid {} argument, frequency should be in form of '1m', '5m', '1d', '1w' got {} (type: {})").format(func_name, self.arg_name, value, type(value)))

    def is_valid_frequency(self):
        if False:
            for i in range(10):
                print('nop')
        self._rules.append(self._is_valid_frequency)
        return self

    def verify(self, func_name, call_args):
        if False:
            print('Hello World!')
        value = call_args[self.arg_name]
        for r in self._rules:
            r(func_name, value)

    @property
    def arg_name(self):
        if False:
            i = 10
            return i + 15
        return self._arg_name

    @property
    def pre_check(self):
        if False:
            return 10
        return self._pre_check

def verify_that(arg_name, pre_check=False):
    if False:
        while True:
            i = 10
    return ArgumentChecker(arg_name, pre_check)

def get_call_args(func, args, kwargs, traceback=None):
    if False:
        return 10
    try:
        return inspect.getcallargs(unwrapper(func), *args, **kwargs)
    except TypeError as e:
        six.reraise(RQTypeError, RQTypeError(*e.args), traceback)

class ApiArgumentsChecker(object):

    def __init__(self, rules):
        if False:
            return 10
        self._rules = OrderedDict()
        for r in rules:
            self._rules[r.arg_name] = r

    @property
    def rules(self):
        if False:
            i = 10
            return i + 15
        return self._rules

    @property
    def pre_check_rules(self):
        if False:
            return 10
        for r in six.itervalues(self._rules):
            if r.pre_check:
                yield r

    @property
    def post_check_rules(self):
        if False:
            i = 10
            return i + 15
        for r in six.itervalues(self._rules):
            if not r.pre_check:
                yield r

    @contextmanager
    def check(self, func, args, kwargs):
        if False:
            print('Hello World!')
        call_args = None
        for r in self.pre_check_rules:
            if call_args is None:
                call_args = get_call_args(func, args, kwargs)
            r.verify(func.__name__, call_args)
        try:
            yield
        except RQInvalidArgument:
            raise
        except Exception as e:
            exc_info = sys.exc_info()
            (t, v, tb) = exc_info
            if call_args is None:
                call_args = get_call_args(func, args, kwargs, tb)
            try:
                for r in self.post_check_rules:
                    r.verify(func.__name__, call_args)
            except RQInvalidArgument as e:
                six.reraise(RQInvalidArgument, e, tb)
                return
            if getattr(e, EXC_EXT_NAME, EXC_TYPE.NOTSET) == EXC_TYPE.NOTSET:
                patch_system_exc(e)
            raise

def apply_rules(*rules):
    if False:
        print('Hello World!')
    checker = ApiArgumentsChecker(rules)

    def decorator(func):
        if False:
            print('Hello World!')

        @wraps(func)
        def api_rule_check_wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            with checker.check(func, args, kwargs):
                return func(*args, **kwargs)
        api_rule_check_wrapper._rq_api_args_checker = checker
        api_rule_check_wrapper._rq_exception_checked = True
        return api_rule_check_wrapper
    return decorator