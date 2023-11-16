from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import collections
import itertools
import multiprocessing
try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections
import backtrader as bt
from .utils.py3 import map, range, zip, with_metaclass, string_types, integer_types
from . import linebuffer
from . import indicator
from .brokers import BackBroker
from .metabase import MetaParams
from . import observers
from .writer import WriterFile
from .utils import OrderedDict, tzparse, num2date, date2num
from .strategy import Strategy, SignalStrategy
from .tradingcal import TradingCalendarBase, TradingCalendar, PandasMarketCalendar
from .timer import Timer

class OptReturn(object):

    def __init__(self, params, **kwargs):
        if False:
            i = 10
            return i + 15
        self.p = self.params = params
        for (k, v) in kwargs.items():
            setattr(self, k, v)

class Cerebro(with_metaclass(MetaParams, object)):
    """Params:

      - ``preload`` (default: ``True``)

        Whether to preload the different ``data feeds`` passed to cerebro for
        the Strategies

      - ``runonce`` (default: ``True``)

        Run ``Indicators`` in vectorized mode to speed up the entire system.
        Strategies and Observers will always be run on an event based basis

      - ``live`` (default: ``False``)

        If no data has reported itself as *live* (via the data's ``islive``
        method but the end user still want to run in ``live`` mode, this
        parameter can be set to true

        This will simultaneously deactivate ``preload`` and ``runonce``. It
        will have no effect on memory saving schemes.

        Run ``Indicators`` in vectorized mode to speed up the entire system.
        Strategies and Observers will always be run on an event based basis

      - ``maxcpus`` (default: None -> all available cores)

         How many cores to use simultaneously for optimization

      - ``stdstats`` (default: ``True``)

        If True default Observers will be added: Broker (Cash and Value),
        Trades and BuySell

      - ``oldbuysell`` (default: ``False``)

        If ``stdstats`` is ``True`` and observers are getting automatically
        added, this switch controls the main behavior of the ``BuySell``
        observer

        - ``False``: use the modern behavior in which the buy / sell signals
          are plotted below / above the low / high prices respectively to avoid
          cluttering the plot

        - ``True``: use the deprecated behavior in which the buy / sell signals
          are plotted where the average price of the order executions for the
          given moment in time is. This will of course be on top of an OHLC bar
          or on a Line on Cloe bar, difficulting the recognition of the plot.

      - ``oldtrades`` (default: ``False``)

        If ``stdstats`` is ``True`` and observers are getting automatically
        added, this switch controls the main behavior of the ``Trades``
        observer

        - ``False``: use the modern behavior in which trades for all datas are
          plotted with different markers

        - ``True``: use the old Trades observer which plots the trades with the
          same markers, differentiating only if they are positive or negative

      - ``exactbars`` (default: ``False``)

        With the default value each and every value stored in a line is kept in
        memory

        Possible values:
          - ``True`` or ``1``: all "lines" objects reduce memory usage to the
            automatically calculated minimum period.

            If a Simple Moving Average has a period of 30, the underlying data
            will have always a running buffer of 30 bars to allow the
            calculation of the Simple Moving Average

            - This setting will deactivate ``preload`` and ``runonce``
            - Using this setting also deactivates **plotting**

          - ``-1``: datafreeds and indicators/operations at strategy level will
            keep all data in memory.

            For example: a ``RSI`` internally uses the indicator ``UpDay`` to
            make calculations. This subindicator will not keep all data in
            memory

            - This allows to keep ``plotting`` and ``preloading`` active.

            - ``runonce`` will be deactivated

          - ``-2``: data feeds and indicators kept as attributes of the
            strategy will keep all points in memory.

            For example: a ``RSI`` internally uses the indicator ``UpDay`` to
            make calculations. This subindicator will not keep all data in
            memory

            If in the ``__init__`` something like
            ``a = self.data.close - self.data.high`` is defined, then ``a``
            will not keep all data in memory

            - This allows to keep ``plotting`` and ``preloading`` active.

            - ``runonce`` will be deactivated

      - ``objcache`` (default: ``False``)

        Experimental option to implement a cache of lines objects and reduce
        the amount of them. Example from UltimateOscillator::

          bp = self.data.close - TrueLow(self.data)
          tr = TrueRange(self.data)  # -> creates another TrueLow(self.data)

        If this is ``True`` the 2nd ``TrueLow(self.data)`` inside ``TrueRange``
        matches the signature of the one in the ``bp`` calculation. It will be
        reused.

        Corner cases may happen in which this drives a line object off its
        minimum period and breaks things and it is therefore disabled.

      - ``writer`` (default: ``False``)

        If set to ``True`` a default WriterFile will be created which will
        print to stdout. It will be added to the strategy (in addition to any
        other writers added by the user code)

      - ``tradehistory`` (default: ``False``)

        If set to ``True``, it will activate update event logging in each trade
        for all strategies. This can also be accomplished on a per strategy
        basis with the strategy method ``set_tradehistory``

      - ``optdatas`` (default: ``True``)

        If ``True`` and optimizing (and the system can ``preload`` and use
        ``runonce``, data preloading will be done only once in the main process
        to save time and resources.

        The tests show an approximate ``20%`` speed-up moving from a sample
        execution in ``83`` seconds to ``66``

      - ``optreturn`` (default: ``True``)

        If ``True`` the optimization results will not be full ``Strategy``
        objects (and all *datas*, *indicators*, *observers* ...) but and object
        with the following attributes (same as in ``Strategy``):

          - ``params`` (or ``p``) the strategy had for the execution
          - ``analyzers`` the strategy has executed

        In most occassions, only the *analyzers* and with which *params* are
        the things needed to evaluate a the performance of a strategy. If
        detailed analysis of the generated values for (for example)
        *indicators* is needed, turn this off

        The tests show a ``13% - 15%`` improvement in execution time. Combined
        with ``optdatas`` the total gain increases to a total speed-up of
        ``32%`` in an optimization run.

      - ``oldsync`` (default: ``False``)

        Starting with release 1.9.0.99 the synchronization of multiple datas
        (same or different timeframes) has been changed to allow datas of
        different lengths.

        If the old behavior with data0 as the master of the system is wished,
        set this parameter to true

      - ``tz`` (default: ``None``)

        Adds a global timezone for strategies. The argument ``tz`` can be

          - ``None``: in this case the datetime displayed by strategies will be
            in UTC, which has been always the standard behavior

          - ``pytz`` instance. It will be used as such to convert UTC times to
            the chosen timezone

          - ``string``. Instantiating a ``pytz`` instance will be attempted.

          - ``integer``. Use, for the strategy, the same timezone as the
            corresponding ``data`` in the ``self.datas`` iterable (``0`` would
            use the timezone from ``data0``)

      - ``cheat_on_open`` (default: ``False``)

        The ``next_open`` method of strategies will be called. This happens
        before ``next`` and before the broker has had a chance to evaluate
        orders. The indicators have not yet been recalculated. This allows
        issuing an orde which takes into account the indicators of the previous
        day but uses the ``open`` price for stake calculations

        For cheat_on_open order execution, it is also necessary to make the
        call ``cerebro.broker.set_coo(True)`` or instantite a broker with
        ``BackBroker(coo=True)`` (where *coo* stands for cheat-on-open) or set
        the ``broker_coo`` parameter to ``True``. Cerebro will do it
        automatically unless disabled below.

      - ``broker_coo`` (default: ``True``)

        This will automatically invoke the ``set_coo`` method of the broker
        with ``True`` to activate ``cheat_on_open`` execution. Will only do it
        if ``cheat_on_open`` is also ``True``

      - ``quicknotify`` (default: ``False``)

        Broker notifications are delivered right before the delivery of the
        *next* prices. For backtesting this has no implications, but with live
        brokers a notification can take place long before the bar is
        delivered. When set to ``True`` notifications will be delivered as soon
        as possible (see ``qcheck`` in live feeds)

        Set to ``False`` for compatibility. May be changed to ``True``

    """
    params = (('preload', True), ('runonce', True), ('maxcpus', None), ('stdstats', True), ('oldbuysell', False), ('oldtrades', False), ('lookahead', 0), ('exactbars', False), ('optdatas', True), ('optreturn', True), ('objcache', False), ('live', False), ('writer', False), ('tradehistory', False), ('oldsync', False), ('tz', None), ('cheat_on_open', False), ('broker_coo', True), ('quicknotify', False))

    def __init__(self):
        if False:
            print('Hello World!')
        self._dolive = False
        self._doreplay = False
        self._dooptimize = False
        self.stores = list()
        self.feeds = list()
        self.datas = list()
        self.datasbyname = collections.OrderedDict()
        self.strats = list()
        self.optcbs = list()
        self.observers = list()
        self.analyzers = list()
        self.indicators = list()
        self.sizers = dict()
        self.writers = list()
        self.storecbs = list()
        self.datacbs = list()
        self.signals = list()
        self._signal_strat = (None, None, None)
        self._signal_concurrent = False
        self._signal_accumulate = False
        self._dataid = itertools.count(1)
        self._broker = BackBroker()
        self._broker.cerebro = self
        self._tradingcal = None
        self._pretimers = list()
        self._ohistory = list()
        self._fhistory = None

    @staticmethod
    def iterize(iterable):
        if False:
            return 10
        'Handy function which turns things into things that can be iterated upon\n        including iterables\n        '
        niterable = list()
        for elem in iterable:
            if isinstance(elem, string_types):
                elem = (elem,)
            elif not isinstance(elem, collectionsAbc.Iterable):
                elem = (elem,)
            niterable.append(elem)
        return niterable

    def set_fund_history(self, fund):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a history of orders to be directly executed in the broker for\n        performance evaluation\n\n          - ``fund``: is an iterable (ex: list, tuple, iterator, generator)\n            in which each element will be also an iterable (with length) with\n            the following sub-elements (2 formats are possible)\n\n            ``[datetime, share_value, net asset value]``\n\n            **Note**: it must be sorted (or produce sorted elements) by\n              datetime ascending\n\n            where:\n\n              - ``datetime`` is a python ``date/datetime`` instance or a string\n                with format YYYY-MM-DD[THH:MM:SS[.us]] where the elements in\n                brackets are optional\n              - ``share_value`` is an float/integer\n              - ``net_asset_value`` is a float/integer\n        '
        self._fhistory = fund

    def add_order_history(self, orders, notify=True):
        if False:
            return 10
        '\n        Add a history of orders to be directly executed in the broker for\n        performance evaluation\n\n          - ``orders``: is an iterable (ex: list, tuple, iterator, generator)\n            in which each element will be also an iterable (with length) with\n            the following sub-elements (2 formats are possible)\n\n            ``[datetime, size, price]`` or ``[datetime, size, price, data]``\n\n            **Note**: it must be sorted (or produce sorted elements) by\n              datetime ascending\n\n            where:\n\n              - ``datetime`` is a python ``date/datetime`` instance or a string\n                with format YYYY-MM-DD[THH:MM:SS[.us]] where the elements in\n                brackets are optional\n              - ``size`` is an integer (positive to *buy*, negative to *sell*)\n              - ``price`` is a float/integer\n              - ``data`` if present can take any of the following values\n\n                - *None* - The 1st data feed will be used as target\n                - *integer* - The data with that index (insertion order in\n                  **Cerebro**) will be used\n                - *string* - a data with that name, assigned for example with\n                  ``cerebro.addata(data, name=value)``, will be the target\n\n          - ``notify`` (default: *True*)\n\n            If ``True`` the 1st strategy inserted in the system will be\n            notified of the artificial orders created following the information\n            from each order in ``orders``\n\n        **Note**: Implicit in the description is the need to add a data feed\n          which is the target of the orders. This is for example needed by\n          analyzers which track for example the returns\n        '
        self._ohistory.append((orders, notify))

    def notify_timer(self, timer, when, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Receives a timer notification where ``timer`` is the timer which was\n        returned by ``add_timer``, and ``when`` is the calling time. ``args``\n        and ``kwargs`` are any additional arguments passed to ``add_timer``\n\n        The actual ``when`` time can be later, but the system may have not be\n        able to call the timer before. This value is the timer value and no the\n        system time.\n        '
        pass

    def _add_timer(self, owner, when, offset=datetime.timedelta(), repeat=datetime.timedelta(), weekdays=[], weekcarry=False, monthdays=[], monthcarry=True, allow=None, tzdata=None, strats=False, cheat=False, *args, **kwargs):
        if False:
            return 10
        'Internal method to really create the timer (not started yet) which\n        can be called by cerebro instances or other objects which can access\n        cerebro'
        timer = Timer(*args, tid=len(self._pretimers), owner=owner, strats=strats, when=when, offset=offset, repeat=repeat, weekdays=weekdays, weekcarry=weekcarry, monthdays=monthdays, monthcarry=monthcarry, allow=allow, tzdata=tzdata, cheat=cheat, **kwargs)
        self._pretimers.append(timer)
        return timer

    def add_timer(self, when, offset=datetime.timedelta(), repeat=datetime.timedelta(), weekdays=[], weekcarry=False, monthdays=[], monthcarry=True, allow=None, tzdata=None, strats=False, cheat=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Schedules a timer to invoke ``notify_timer``\n\n        Arguments:\n\n          - ``when``: can be\n\n            - ``datetime.time`` instance (see below ``tzdata``)\n            - ``bt.timer.SESSION_START`` to reference a session start\n            - ``bt.timer.SESSION_END`` to reference a session end\n\n         - ``offset`` which must be a ``datetime.timedelta`` instance\n\n           Used to offset the value ``when``. It has a meaningful use in\n           combination with ``SESSION_START`` and ``SESSION_END``, to indicated\n           things like a timer being called ``15 minutes`` after the session\n           start.\n\n          - ``repeat`` which must be a ``datetime.timedelta`` instance\n\n            Indicates if after a 1st call, further calls will be scheduled\n            within the same session at the scheduled ``repeat`` delta\n\n            Once the timer goes over the end of the session it is reset to the\n            original value for ``when``\n\n          - ``weekdays``: a **sorted** iterable with integers indicating on\n            which days (iso codes, Monday is 1, Sunday is 7) the timers can\n            be actually invoked\n\n            If not specified, the timer will be active on all days\n\n          - ``weekcarry`` (default: ``False``). If ``True`` and the weekday was\n            not seen (ex: trading holiday), the timer will be executed on the\n            next day (even if in a new week)\n\n          - ``monthdays``: a **sorted** iterable with integers indicating on\n            which days of the month a timer has to be executed. For example\n            always on day *15* of the month\n\n            If not specified, the timer will be active on all days\n\n          - ``monthcarry`` (default: ``True``). If the day was not seen\n            (weekend, trading holiday), the timer will be executed on the next\n            available day.\n\n          - ``allow`` (default: ``None``). A callback which receives a\n            `datetime.date`` instance and returns ``True`` if the date is\n            allowed for timers or else returns ``False``\n\n          - ``tzdata`` which can be either ``None`` (default), a ``pytz``\n            instance or a ``data feed`` instance.\n\n            ``None``: ``when`` is interpreted at face value (which translates\n            to handling it as if it where UTC even if it's not)\n\n            ``pytz`` instance: ``when`` will be interpreted as being specified\n            in the local time specified by the timezone instance.\n\n            ``data feed`` instance: ``when`` will be interpreted as being\n            specified in the local time specified by the ``tz`` parameter of\n            the data feed instance.\n\n            **Note**: If ``when`` is either ``SESSION_START`` or\n              ``SESSION_END`` and ``tzdata`` is ``None``, the 1st *data feed*\n              in the system (aka ``self.data0``) will be used as the reference\n              to find out the session times.\n\n          - ``strats`` (default: ``False``) call also the ``notify_timer`` of\n            strategies\n\n          - ``cheat`` (default ``False``) if ``True`` the timer will be called\n            before the broker has a chance to evaluate the orders. This opens\n            the chance to issue orders based on opening price for example right\n            before the session starts\n          - ``*args``: any extra args will be passed to ``notify_timer``\n\n          - ``**kwargs``: any extra kwargs will be passed to ``notify_timer``\n\n        Return Value:\n\n          - The created timer\n\n        "
        return self._add_timer(*args, owner=self, when=when, offset=offset, repeat=repeat, weekdays=weekdays, weekcarry=weekcarry, monthdays=monthdays, monthcarry=monthcarry, allow=allow, tzdata=tzdata, strats=strats, cheat=cheat, **kwargs)

    def addtz(self, tz):
        if False:
            while True:
                i = 10
        '\n        This can also be done with the parameter ``tz``\n\n        Adds a global timezone for strategies. The argument ``tz`` can be\n\n          - ``None``: in this case the datetime displayed by strategies will be\n            in UTC, which has been always the standard behavior\n\n          - ``pytz`` instance. It will be used as such to convert UTC times to\n            the chosen timezone\n\n          - ``string``. Instantiating a ``pytz`` instance will be attempted.\n\n          - ``integer``. Use, for the strategy, the same timezone as the\n            corresponding ``data`` in the ``self.datas`` iterable (``0`` would\n            use the timezone from ``data0``)\n\n        '
        self.p.tz = tz

    def addcalendar(self, cal):
        if False:
            i = 10
            return i + 15
        'Adds a global trading calendar to the system. Individual data feeds\n        may have separate calendars which override the global one\n\n        ``cal`` can be an instance of ``TradingCalendar`` a string or an\n        instance of ``pandas_market_calendars``. A string will be will be\n        instantiated as a ``PandasMarketCalendar`` (which needs the module\n        ``pandas_market_calendar`` installed in the system.\n\n        If a subclass of `TradingCalendarBase` is passed (not an instance) it\n        will be instantiated\n        '
        if isinstance(cal, string_types):
            cal = PandasMarketCalendar(calendar=cal)
        elif hasattr(cal, 'valid_days'):
            cal = PandasMarketCalendar(calendar=cal)
        else:
            try:
                if issubclass(cal, TradingCalendarBase):
                    cal = cal()
            except TypeError:
                pass
        self._tradingcal = cal

    def add_signal(self, sigtype, sigcls, *sigargs, **sigkwargs):
        if False:
            print('Hello World!')
        'Adds a signal to the system which will be later added to a\n        ``SignalStrategy``'
        self.signals.append((sigtype, sigcls, sigargs, sigkwargs))

    def signal_strategy(self, stratcls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Adds a SignalStrategy subclass which can accept signals'
        self._signal_strat = (stratcls, args, kwargs)

    def signal_concurrent(self, onoff):
        if False:
            i = 10
            return i + 15
        'If signals are added to the system and the ``concurrent`` value is\n        set to True, concurrent orders will be allowed'
        self._signal_concurrent = onoff

    def signal_accumulate(self, onoff):
        if False:
            for i in range(10):
                print('nop')
        'If signals are added to the system and the ``accumulate`` value is\n        set to True, entering the market when already in the market, will be\n        allowed to increase a position'
        self._signal_accumulate = onoff

    def addstore(self, store):
        if False:
            return 10
        'Adds an ``Store`` instance to the if not already present'
        if store not in self.stores:
            self.stores.append(store)

    def addwriter(self, wrtcls, *args, **kwargs):
        if False:
            print('Hello World!')
        'Adds an ``Writer`` class to the mix. Instantiation will be done at\n        ``run`` time in cerebro\n        '
        self.writers.append((wrtcls, args, kwargs))

    def addsizer(self, sizercls, *args, **kwargs):
        if False:
            return 10
        'Adds a ``Sizer`` class (and args) which is the default sizer for any\n        strategy added to cerebro\n        '
        self.sizers[None] = (sizercls, args, kwargs)

    def addsizer_byidx(self, idx, sizercls, *args, **kwargs):
        if False:
            print('Hello World!')
        'Adds a ``Sizer`` class by idx. This idx is a reference compatible to\n        the one returned by ``addstrategy``. Only the strategy referenced by\n        ``idx`` will receive this size\n        '
        self.sizers[idx] = (sizercls, args, kwargs)

    def addindicator(self, indcls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds an ``Indicator`` class to the mix. Instantiation will be done at\n        ``run`` time in the passed strategies\n        '
        self.indicators.append((indcls, args, kwargs))

    def addanalyzer(self, ancls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds an ``Analyzer`` class to the mix. Instantiation will be done at\n        ``run`` time\n        '
        self.analyzers.append((ancls, args, kwargs))

    def addobserver(self, obscls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds an ``Observer`` class to the mix. Instantiation will be done at\n        ``run`` time\n        '
        self.observers.append((False, obscls, args, kwargs))

    def addobservermulti(self, obscls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds an ``Observer`` class to the mix. Instantiation will be done at\n        ``run`` time\n\n        It will be added once per "data" in the system. A use case is a\n        buy/sell observer which observes individual datas.\n\n        A counter-example is the CashValue, which observes system-wide values\n        '
        self.observers.append((True, obscls, args, kwargs))

    def addstorecb(self, callback):
        if False:
            for i in range(10):
                print('nop')
        'Adds a callback to get messages which would be handled by the\n        notify_store method\n\n        The signature of the callback must support the following:\n\n          - callback(msg, \\*args, \\*\\*kwargs)\n\n        The actual ``msg``, ``*args`` and ``**kwargs`` received are\n        implementation defined (depend entirely on the *data/broker/store*) but\n        in general one should expect them to be *printable* to allow for\n        reception and experimentation.\n        '
        self.storecbs.append(callback)

    def _notify_store(self, msg, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        for callback in self.storecbs:
            callback(msg, *args, **kwargs)
        self.notify_store(msg, *args, **kwargs)

    def notify_store(self, msg, *args, **kwargs):
        if False:
            print('Hello World!')
        'Receive store notifications in cerebro\n\n        This method can be overridden in ``Cerebro`` subclasses\n\n        The actual ``msg``, ``*args`` and ``**kwargs`` received are\n        implementation defined (depend entirely on the *data/broker/store*) but\n        in general one should expect them to be *printable* to allow for\n        reception and experimentation.\n        '
        pass

    def _storenotify(self):
        if False:
            while True:
                i = 10
        for store in self.stores:
            for notif in store.get_notifications():
                (msg, args, kwargs) = notif
                self._notify_store(msg, *args, **kwargs)
                for strat in self.runningstrats:
                    strat.notify_store(msg, *args, **kwargs)

    def adddatacb(self, callback):
        if False:
            print('Hello World!')
        'Adds a callback to get messages which would be handled by the\n        notify_data method\n\n        The signature of the callback must support the following:\n\n          - callback(data, status, \\*args, \\*\\*kwargs)\n\n        The actual ``*args`` and ``**kwargs`` received are implementation\n        defined (depend entirely on the *data/broker/store*) but in general one\n        should expect them to be *printable* to allow for reception and\n        experimentation.\n        '
        self.datacbs.append(callback)

    def _datanotify(self):
        if False:
            for i in range(10):
                print('nop')
        for data in self.datas:
            for notif in data.get_notifications():
                (status, args, kwargs) = notif
                self._notify_data(data, status, *args, **kwargs)
                for strat in self.runningstrats:
                    strat.notify_data(data, status, *args, **kwargs)

    def _notify_data(self, data, status, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        for callback in self.datacbs:
            callback(data, status, *args, **kwargs)
        self.notify_data(data, status, *args, **kwargs)

    def notify_data(self, data, status, *args, **kwargs):
        if False:
            print('Hello World!')
        'Receive data notifications in cerebro\n\n        This method can be overridden in ``Cerebro`` subclasses\n\n        The actual ``*args`` and ``**kwargs`` received are\n        implementation defined (depend entirely on the *data/broker/store*) but\n        in general one should expect them to be *printable* to allow for\n        reception and experimentation.\n        '
        pass

    def adddata(self, data, name=None):
        if False:
            while True:
                i = 10
        '\n        Adds a ``Data Feed`` instance to the mix.\n\n        If ``name`` is not None it will be put into ``data._name`` which is\n        meant for decoration/plotting purposes.\n        '
        if name is not None:
            data._name = name
        data._id = next(self._dataid)
        data.setenvironment(self)
        self.datas.append(data)
        self.datasbyname[data._name] = data
        feed = data.getfeed()
        if feed and feed not in self.feeds:
            self.feeds.append(feed)
        if data.islive():
            self._dolive = True
        return data

    def chaindata(self, *args, **kwargs):
        if False:
            return 10
        '\n        Chains several data feeds into one\n\n        If ``name`` is passed as named argument and is not None it will be put\n        into ``data._name`` which is meant for decoration/plotting purposes.\n\n        If ``None``, then the name of the 1st data will be used\n        '
        dname = kwargs.pop('name', None)
        if dname is None:
            dname = args[0]._dataname
        d = bt.feeds.Chainer(*args, dataname=dname)
        self.adddata(d, name=dname)
        return d

    def rolloverdata(self, *args, **kwargs):
        if False:
            return 10
        'Chains several data feeds into one\n\n        If ``name`` is passed as named argument and is not None it will be put\n        into ``data._name`` which is meant for decoration/plotting purposes.\n\n        If ``None``, then the name of the 1st data will be used\n\n        Any other kwargs will be passed to the RollOver class\n\n        '
        dname = kwargs.pop('name', None)
        if dname is None:
            dname = args[0]._dataname
        d = bt.feeds.RollOver(*args, dataname=dname, **kwargs)
        self.adddata(d, name=dname)
        return d

    def replaydata(self, dataname, name=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Adds a ``Data Feed`` to be replayed by the system\n\n        If ``name`` is not None it will be put into ``data._name`` which is\n        meant for decoration/plotting purposes.\n\n        Any other kwargs like ``timeframe``, ``compression``, ``todate`` which\n        are supported by the replay filter will be passed transparently\n        '
        if any((dataname is x for x in self.datas)):
            dataname = dataname.clone()
        dataname.replay(**kwargs)
        self.adddata(dataname, name=name)
        self._doreplay = True
        return dataname

    def resampledata(self, dataname, name=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Adds a ``Data Feed`` to be resample by the system\n\n        If ``name`` is not None it will be put into ``data._name`` which is\n        meant for decoration/plotting purposes.\n\n        Any other kwargs like ``timeframe``, ``compression``, ``todate`` which\n        are supported by the resample filter will be passed transparently\n        '
        if any((dataname is x for x in self.datas)):
            dataname = dataname.clone()
        dataname.resample(**kwargs)
        self.adddata(dataname, name=name)
        self._doreplay = True
        return dataname

    def optcallback(self, cb):
        if False:
            while True:
                i = 10
        '\n        Adds a *callback* to the list of callbacks that will be called with the\n        optimizations when each of the strategies has been run\n\n        The signature: cb(strategy)\n        '
        self.optcbs.append(cb)

    def optstrategy(self, strategy, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Adds a ``Strategy`` class to the mix for optimization. Instantiation\n        will happen during ``run`` time.\n\n        args and kwargs MUST BE iterables which hold the values to check.\n\n        Example: if a Strategy accepts a parameter ``period``, for optimization\n        purposes the call to ``optstrategy`` looks like:\n\n          - cerebro.optstrategy(MyStrategy, period=(15, 25))\n\n        This will execute an optimization for values 15 and 25. Whereas\n\n          - cerebro.optstrategy(MyStrategy, period=range(15, 25))\n\n        will execute MyStrategy with ``period`` values 15 -> 25 (25 not\n        included, because ranges are semi-open in Python)\n\n        If a parameter is passed but shall not be optimized the call looks\n        like:\n\n          - cerebro.optstrategy(MyStrategy, period=(15,))\n\n        Notice that ``period`` is still passed as an iterable ... of just 1\n        element\n\n        ``backtrader`` will anyhow try to identify situations like:\n\n          - cerebro.optstrategy(MyStrategy, period=15)\n\n        and will create an internal pseudo-iterable if possible\n        '
        self._dooptimize = True
        args = self.iterize(args)
        optargs = itertools.product(*args)
        optkeys = list(kwargs)
        vals = self.iterize(kwargs.values())
        optvals = itertools.product(*vals)
        okwargs1 = map(zip, itertools.repeat(optkeys), optvals)
        optkwargs = map(dict, okwargs1)
        it = itertools.product([strategy], optargs, optkwargs)
        self.strats.append(it)

    def addstrategy(self, strategy, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a ``Strategy`` class to the mix for a single pass run.\n        Instantiation will happen during ``run`` time.\n\n        args and kwargs will be passed to the strategy as they are during\n        instantiation.\n\n        Returns the index with which addition of other objects (like sizers)\n        can be referenced\n        '
        self.strats.append([(strategy, args, kwargs)])
        return len(self.strats) - 1

    def setbroker(self, broker):
        if False:
            i = 10
            return i + 15
        '\n        Sets a specific ``broker`` instance for this strategy, replacing the\n        one inherited from cerebro.\n        '
        self._broker = broker
        broker.cerebro = self
        return broker

    def getbroker(self):
        if False:
            while True:
                i = 10
        '\n        Returns the broker instance.\n\n        This is also available as a ``property`` by the name ``broker``\n        '
        return self._broker
    broker = property(getbroker, setbroker)

    def plot(self, plotter=None, numfigs=1, iplot=True, start=None, end=None, width=16, height=9, dpi=300, tight=True, use=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Plots the strategies inside cerebro\n\n        If ``plotter`` is None a default ``Plot`` instance is created and\n        ``kwargs`` are passed to it during instantiation.\n\n        ``numfigs`` split the plot in the indicated number of charts reducing\n        chart density if wished\n\n        ``iplot``: if ``True`` and running in a ``notebook`` the charts will be\n        displayed inline\n\n        ``use``: set it to the name of the desired matplotlib backend. It will\n        take precedence over ``iplot``\n\n        ``start``: An index to the datetime line array of the strategy or a\n        ``datetime.date``, ``datetime.datetime`` instance indicating the start\n        of the plot\n\n        ``end``: An index to the datetime line array of the strategy or a\n        ``datetime.date``, ``datetime.datetime`` instance indicating the end\n        of the plot\n\n        ``width``: in inches of the saved figure\n\n        ``height``: in inches of the saved figure\n\n        ``dpi``: quality in dots per inches of the saved figure\n\n        ``tight``: only save actual content and not the frame of the figure\n        '
        if self._exactbars > 0:
            return
        if not plotter:
            from . import plot
            if self.p.oldsync:
                plotter = plot.Plot_OldSync(**kwargs)
            else:
                plotter = plot.Plot(**kwargs)
        figs = []
        for stratlist in self.runstrats:
            for (si, strat) in enumerate(stratlist):
                rfig = plotter.plot(strat, figid=si * 100, numfigs=numfigs, iplot=iplot, start=start, end=end, use=use)
                figs.append(rfig)
            plotter.show()
        return figs

    def __call__(self, iterstrat):
        if False:
            for i in range(10):
                print('nop')
        '\n        Used during optimization to pass the cerebro over the multiprocesing\n        module without complains\n        '
        predata = self.p.optdatas and self._dopreload and self._dorunonce
        return self.runstrategies(iterstrat, predata=predata)

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Used during optimization to prevent optimization result `runstrats`\n        from being pickled to subprocesses\n        '
        rv = vars(self).copy()
        if 'runstrats' in rv:
            del rv['runstrats']
        return rv

    def runstop(self):
        if False:
            while True:
                i = 10
        'If invoked from inside a strategy or anywhere else, including other\n        threads the execution will stop as soon as possible.'
        self._event_stop = True

    def run(self, **kwargs):
        if False:
            while True:
                i = 10
        'The core method to perform backtesting. Any ``kwargs`` passed to it\n        will affect the value of the standard parameters ``Cerebro`` was\n        instantiated with.\n\n        If ``cerebro`` has not datas the method will immediately bail out.\n\n        It has different return values:\n\n          - For No Optimization: a list contanining instances of the Strategy\n            classes added with ``addstrategy``\n\n          - For Optimization: a list of lists which contain instances of the\n            Strategy classes added with ``addstrategy``\n        '
        self._event_stop = False
        if not self.datas:
            return []
        pkeys = self.params._getkeys()
        for (key, val) in kwargs.items():
            if key in pkeys:
                setattr(self.params, key, val)
        linebuffer.LineActions.cleancache()
        indicator.Indicator.cleancache()
        linebuffer.LineActions.usecache(self.p.objcache)
        indicator.Indicator.usecache(self.p.objcache)
        self._dorunonce = self.p.runonce
        self._dopreload = self.p.preload
        self._exactbars = int(self.p.exactbars)
        if self._exactbars:
            self._dorunonce = False
            self._dopreload = self._dopreload and self._exactbars < 1
        self._doreplay = self._doreplay or any((x.replaying for x in self.datas))
        if self._doreplay:
            self._dopreload = False
        if self._dolive or self.p.live:
            self._dorunonce = False
            self._dopreload = False
        self.runwriters = list()
        if self.p.writer is True:
            wr = WriterFile()
            self.runwriters.append(wr)
        for (wrcls, wrargs, wrkwargs) in self.writers:
            wr = wrcls(*wrargs, **wrkwargs)
            self.runwriters.append(wr)
        self.writers_csv = any(map(lambda x: x.p.csv, self.runwriters))
        self.runstrats = list()
        if self.signals:
            (signalst, sargs, skwargs) = self._signal_strat
            if signalst is None:
                try:
                    (signalst, sargs, skwargs) = self.strats.pop(0)
                except IndexError:
                    pass
                else:
                    if not isinstance(signalst, SignalStrategy):
                        self.strats.insert(0, (signalst, sargs, skwargs))
                        signalst = None
            if signalst is None:
                (signalst, sargs, skwargs) = (SignalStrategy, tuple(), dict())
            self.addstrategy(signalst, *sargs, _accumulate=self._signal_accumulate, _concurrent=self._signal_concurrent, signals=self.signals, **skwargs)
        if not self.strats:
            self.addstrategy(Strategy)
        iterstrats = itertools.product(*self.strats)
        if not self._dooptimize or self.p.maxcpus == 1:
            for iterstrat in iterstrats:
                runstrat = self.runstrategies(iterstrat)
                self.runstrats.append(runstrat)
                if self._dooptimize:
                    for cb in self.optcbs:
                        cb(runstrat)
        else:
            if self.p.optdatas and self._dopreload and self._dorunonce:
                for data in self.datas:
                    data.reset()
                    if self._exactbars < 1:
                        data.extend(size=self.params.lookahead)
                    data._start()
                    if self._dopreload:
                        data.preload()
            pool = multiprocessing.Pool(self.p.maxcpus or None)
            for r in pool.imap(self, iterstrats):
                self.runstrats.append(r)
                for cb in self.optcbs:
                    cb(r)
            pool.close()
            if self.p.optdatas and self._dopreload and self._dorunonce:
                for data in self.datas:
                    data.stop()
        if not self._dooptimize:
            return self.runstrats[0]
        return self.runstrats

    def _init_stcount(self):
        if False:
            return 10
        self.stcount = itertools.count(0)

    def _next_stid(self):
        if False:
            for i in range(10):
                print('nop')
        return next(self.stcount)

    def runstrategies(self, iterstrat, predata=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal method invoked by ``run``` to run a set of strategies\n        '
        self._init_stcount()
        self.runningstrats = runstrats = list()
        for store in self.stores:
            store.start()
        if self.p.cheat_on_open and self.p.broker_coo:
            if hasattr(self._broker, 'set_coo'):
                self._broker.set_coo(True)
        if self._fhistory is not None:
            self._broker.set_fund_history(self._fhistory)
        for (orders, onotify) in self._ohistory:
            self._broker.add_order_history(orders, onotify)
        self._broker.start()
        for feed in self.feeds:
            feed.start()
        if self.writers_csv:
            wheaders = list()
            for data in self.datas:
                if data.csv:
                    wheaders.extend(data.getwriterheaders())
            for writer in self.runwriters:
                if writer.p.csv:
                    writer.addheaders(wheaders)
        if not predata:
            for data in self.datas:
                data.reset()
                if self._exactbars < 1:
                    data.extend(size=self.params.lookahead)
                data._start()
                if self._dopreload:
                    data.preload()
        for (stratcls, sargs, skwargs) in iterstrat:
            sargs = self.datas + list(sargs)
            try:
                strat = stratcls(*sargs, **skwargs)
            except bt.errors.StrategySkipError:
                continue
            if self.p.oldsync:
                strat._oldsync = True
            if self.p.tradehistory:
                strat.set_tradehistory()
            runstrats.append(strat)
        tz = self.p.tz
        if isinstance(tz, integer_types):
            tz = self.datas[tz]._tz
        else:
            tz = tzparse(tz)
        if runstrats:
            defaultsizer = self.sizers.get(None, (None, None, None))
            for (idx, strat) in enumerate(runstrats):
                if self.p.stdstats:
                    strat._addobserver(False, observers.Broker)
                    if self.p.oldbuysell:
                        strat._addobserver(True, observers.BuySell)
                    else:
                        strat._addobserver(True, observers.BuySell, barplot=True)
                    if self.p.oldtrades or len(self.datas) == 1:
                        strat._addobserver(False, observers.Trades)
                    else:
                        strat._addobserver(False, observers.DataTrades)
                for (multi, obscls, obsargs, obskwargs) in self.observers:
                    strat._addobserver(multi, obscls, *obsargs, **obskwargs)
                for (indcls, indargs, indkwargs) in self.indicators:
                    strat._addindicator(indcls, *indargs, **indkwargs)
                for (ancls, anargs, ankwargs) in self.analyzers:
                    strat._addanalyzer(ancls, *anargs, **ankwargs)
                (sizer, sargs, skwargs) = self.sizers.get(idx, defaultsizer)
                if sizer is not None:
                    strat._addsizer(sizer, *sargs, **skwargs)
                strat._settz(tz)
                strat._start()
                for writer in self.runwriters:
                    if writer.p.csv:
                        writer.addheaders(strat.getwriterheaders())
            if not predata:
                for strat in runstrats:
                    strat.qbuffer(self._exactbars, replaying=self._doreplay)
            for writer in self.runwriters:
                writer.start()
            self._timers = []
            self._timerscheat = []
            for timer in self._pretimers:
                timer.start(self.datas[0])
                if timer.params.cheat:
                    self._timerscheat.append(timer)
                else:
                    self._timers.append(timer)
            if self._dopreload and self._dorunonce:
                if self.p.oldsync:
                    self._runonce_old(runstrats)
                else:
                    self._runonce(runstrats)
            elif self.p.oldsync:
                self._runnext_old(runstrats)
            else:
                self._runnext(runstrats)
            for strat in runstrats:
                strat._stop()
        self._broker.stop()
        if not predata:
            for data in self.datas:
                data.stop()
        for feed in self.feeds:
            feed.stop()
        for store in self.stores:
            store.stop()
        self.stop_writers(runstrats)
        if self._dooptimize and self.p.optreturn:
            results = list()
            for strat in runstrats:
                for a in strat.analyzers:
                    a.strategy = None
                    a._parent = None
                    for attrname in dir(a):
                        if attrname.startswith('data'):
                            setattr(a, attrname, None)
                oreturn = OptReturn(strat.params, analyzers=strat.analyzers, strategycls=type(strat))
                results.append(oreturn)
            return results
        return runstrats

    def stop_writers(self, runstrats):
        if False:
            i = 10
            return i + 15
        cerebroinfo = OrderedDict()
        datainfos = OrderedDict()
        for (i, data) in enumerate(self.datas):
            datainfos['Data%d' % i] = data.getwriterinfo()
        cerebroinfo['Datas'] = datainfos
        stratinfos = dict()
        for strat in runstrats:
            stname = strat.__class__.__name__
            stratinfos[stname] = strat.getwriterinfo()
        cerebroinfo['Strategies'] = stratinfos
        for writer in self.runwriters:
            writer.writedict(dict(Cerebro=cerebroinfo))
            writer.stop()

    def _brokernotify(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal method which kicks the broker and delivers any broker\n        notification to the strategy\n        '
        self._broker.next()
        while True:
            order = self._broker.get_notification()
            if order is None:
                break
            owner = order.owner
            if owner is None:
                owner = self.runningstrats[0]
            owner._addnotification(order, quicknotify=self.p.quicknotify)

    def _runnext_old(self, runstrats):
        if False:
            return 10
        '\n        Actual implementation of run in full next mode. All objects have its\n        ``next`` method invoke on each data arrival\n        '
        data0 = self.datas[0]
        d0ret = True
        while d0ret or d0ret is None:
            lastret = False
            self._storenotify()
            if self._event_stop:
                return
            self._datanotify()
            if self._event_stop:
                return
            d0ret = data0.next()
            if d0ret:
                for data in self.datas[1:]:
                    if not data.next(datamaster=data0):
                        data._check(forcedata=data0)
                        data.next(datamaster=data0)
            elif d0ret is None:
                data0._check()
                for data in self.datas[1:]:
                    data._check()
            else:
                lastret = data0._last()
                for data in self.datas[1:]:
                    lastret += data._last(datamaster=data0)
                if not lastret:
                    break
            self._datanotify()
            if self._event_stop:
                return
            self._brokernotify()
            if self._event_stop:
                return
            if d0ret or lastret:
                for strat in runstrats:
                    strat._next()
                    if self._event_stop:
                        return
                    self._next_writers(runstrats)
        self._datanotify()
        if self._event_stop:
            return
        self._storenotify()
        if self._event_stop:
            return

    def _runonce_old(self, runstrats):
        if False:
            i = 10
            return i + 15
        '\n        Actual implementation of run in vector mode.\n        Strategies are still invoked on a pseudo-event mode in which ``next``\n        is called for each data arrival\n        '
        for strat in runstrats:
            strat._once()
        data0 = self.datas[0]
        datas = self.datas[1:]
        for i in range(data0.buflen()):
            data0.advance()
            for data in datas:
                data.advance(datamaster=data0)
            self._brokernotify()
            if self._event_stop:
                return
            for strat in runstrats:
                strat._oncepost(data0.datetime[0])
                if self._event_stop:
                    return
                self._next_writers(runstrats)

    def _next_writers(self, runstrats):
        if False:
            print('Hello World!')
        if not self.runwriters:
            return
        if self.writers_csv:
            wvalues = list()
            for data in self.datas:
                if data.csv:
                    wvalues.extend(data.getwritervalues())
            for strat in runstrats:
                wvalues.extend(strat.getwritervalues())
            for writer in self.runwriters:
                if writer.p.csv:
                    writer.addvalues(wvalues)
                    writer.next()

    def _disable_runonce(self):
        if False:
            return 10
        'API for lineiterators to disable runonce (see HeikinAshi)'
        self._dorunonce = False

    def _runnext(self, runstrats):
        if False:
            return 10
        '\n        Actual implementation of run in full next mode. All objects have its\n        ``next`` method invoke on each data arrival\n        '
        datas = sorted(self.datas, key=lambda x: (x._timeframe, x._compression))
        datas1 = datas[1:]
        data0 = datas[0]
        d0ret = True
        rs = [i for (i, x) in enumerate(datas) if x.resampling]
        rp = [i for (i, x) in enumerate(datas) if x.replaying]
        rsonly = [i for (i, x) in enumerate(datas) if x.resampling and (not x.replaying)]
        onlyresample = len(datas) == len(rsonly)
        noresample = not rsonly
        clonecount = sum((d._clone for d in datas))
        ldatas = len(datas)
        ldatas_noclones = ldatas - clonecount
        lastqcheck = False
        dt0 = date2num(datetime.datetime.max) - 2
        while d0ret or d0ret is None:
            newqcheck = not any((d.haslivedata() for d in datas))
            if not newqcheck:
                livecount = sum((d._laststatus == d.LIVE for d in datas))
                newqcheck = not livecount or livecount == ldatas_noclones
            lastret = False
            self._storenotify()
            if self._event_stop:
                return
            self._datanotify()
            if self._event_stop:
                return
            drets = []
            qstart = datetime.datetime.utcnow()
            for d in datas:
                qlapse = datetime.datetime.utcnow() - qstart
                d.do_qcheck(newqcheck, qlapse.total_seconds())
                drets.append(d.next(ticks=False))
            d0ret = any((dret for dret in drets))
            if not d0ret and any((dret is None for dret in drets)):
                d0ret = None
            if d0ret:
                dts = []
                for (i, ret) in enumerate(drets):
                    dts.append(datas[i].datetime[0] if ret else None)
                if onlyresample or noresample:
                    dt0 = min((d for d in dts if d is not None))
                else:
                    dt0 = min((d for (i, d) in enumerate(dts) if d is not None and i not in rsonly))
                dmaster = datas[dts.index(dt0)]
                self._dtmaster = dmaster.num2date(dt0)
                self._udtmaster = num2date(dt0)
                for (i, ret) in enumerate(drets):
                    if ret:
                        continue
                    d = datas[i]
                    d._check(forcedata=dmaster)
                    if d.next(datamaster=dmaster, ticks=False):
                        dts[i] = d.datetime[0]
                    else:
                        pass
                for (i, dti) in enumerate(dts):
                    if dti is not None:
                        di = datas[i]
                        rpi = False and di.replaying
                        if dti > dt0:
                            if not rpi:
                                di.rewind()
                        elif not di.replaying:
                            di._tick_fill(force=True)
            elif d0ret is None:
                for data in datas:
                    data._check()
            else:
                lastret = data0._last()
                for data in datas1:
                    lastret += data._last(datamaster=data0)
                if not lastret:
                    break
            self._datanotify()
            if self._event_stop:
                return
            if d0ret or lastret:
                self._check_timers(runstrats, dt0, cheat=True)
                if self.p.cheat_on_open:
                    for strat in runstrats:
                        strat._next_open()
                        if self._event_stop:
                            return
            self._brokernotify()
            if self._event_stop:
                return
            if d0ret or lastret:
                self._check_timers(runstrats, dt0, cheat=False)
                for strat in runstrats:
                    strat._next()
                    if self._event_stop:
                        return
                    self._next_writers(runstrats)
        self._datanotify()
        if self._event_stop:
            return
        self._storenotify()
        if self._event_stop:
            return

    def _runonce(self, runstrats):
        if False:
            return 10
        '\n        Actual implementation of run in vector mode.\n\n        Strategies are still invoked on a pseudo-event mode in which ``next``\n        is called for each data arrival\n        '
        for strat in runstrats:
            strat._once()
            strat.reset()
        datas = sorted(self.datas, key=lambda x: (x._timeframe, x._compression))
        while True:
            dts = [d.advance_peek() for d in datas]
            dt0 = min(dts)
            if dt0 == float('inf'):
                break
            slen = len(runstrats[0])
            for (i, dti) in enumerate(dts):
                if dti <= dt0:
                    datas[i].advance()
                else:
                    pass
            self._check_timers(runstrats, dt0, cheat=True)
            if self.p.cheat_on_open:
                for strat in runstrats:
                    strat._oncepost_open()
                    if self._event_stop:
                        return
            self._brokernotify()
            if self._event_stop:
                return
            self._check_timers(runstrats, dt0, cheat=False)
            for strat in runstrats:
                strat._oncepost(dt0)
                if self._event_stop:
                    return
                self._next_writers(runstrats)

    def _check_timers(self, runstrats, dt0, cheat=False):
        if False:
            i = 10
            return i + 15
        timers = self._timers if not cheat else self._timerscheat
        for t in timers:
            if not t.check(dt0):
                continue
            t.params.owner.notify_timer(t, t.lastwhen, *t.args, **t.kwargs)
            if t.params.strats:
                for strat in runstrats:
                    strat.notify_timer(t, t.lastwhen, *t.args, **t.kwargs)