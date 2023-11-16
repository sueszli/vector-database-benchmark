"""
.. References and links rendered by Sphinx are kept here as "module documentation" so that they can
   be used in the ``Logger`` docstrings but do not pollute ``help(logger)`` output.

.. |Logger| replace:: :class:`~Logger`
.. |add| replace:: :meth:`~Logger.add()`
.. |remove| replace:: :meth:`~Logger.remove()`
.. |complete| replace:: :meth:`~Logger.complete()`
.. |catch| replace:: :meth:`~Logger.catch()`
.. |bind| replace:: :meth:`~Logger.bind()`
.. |contextualize| replace:: :meth:`~Logger.contextualize()`
.. |patch| replace:: :meth:`~Logger.patch()`
.. |opt| replace:: :meth:`~Logger.opt()`
.. |log| replace:: :meth:`~Logger.log()`
.. |level| replace:: :meth:`~Logger.level()`
.. |enable| replace:: :meth:`~Logger.enable()`
.. |disable| replace:: :meth:`~Logger.disable()`

.. |Any| replace:: :obj:`~typing.Any`
.. |str| replace:: :class:`str`
.. |int| replace:: :class:`int`
.. |bool| replace:: :class:`bool`
.. |tuple| replace:: :class:`tuple`
.. |namedtuple| replace:: :func:`namedtuple<collections.namedtuple>`
.. |list| replace:: :class:`list`
.. |dict| replace:: :class:`dict`
.. |str.format| replace:: :meth:`str.format()`
.. |Path| replace:: :class:`pathlib.Path`
.. |match.groupdict| replace:: :meth:`re.Match.groupdict()`
.. |Handler| replace:: :class:`logging.Handler`
.. |sys.stderr| replace:: :data:`sys.stderr`
.. |sys.exc_info| replace:: :func:`sys.exc_info()`
.. |time| replace:: :class:`datetime.time`
.. |datetime| replace:: :class:`datetime.datetime`
.. |timedelta| replace:: :class:`datetime.timedelta`
.. |open| replace:: :func:`open()`
.. |logging| replace:: :mod:`logging`
.. |signal| replace:: :mod:`signal`
.. |contextvars| replace:: :mod:`contextvars`
.. |multiprocessing| replace:: :mod:`multiprocessing`
.. |Thread.run| replace:: :meth:`Thread.run()<threading.Thread.run()>`
.. |Exception| replace:: :class:`Exception`
.. |AbstractEventLoop| replace:: :class:`AbstractEventLoop<asyncio.AbstractEventLoop>`
.. |asyncio.get_running_loop| replace:: :func:`asyncio.get_running_loop()`
.. |asyncio.run| replace:: :func:`asyncio.run()`
.. |loop.run_until_complete| replace::
    :meth:`loop.run_until_complete()<asyncio.loop.run_until_complete()>`
.. |loop.create_task| replace:: :meth:`loop.create_task()<asyncio.loop.create_task()>`

.. |logger.trace| replace:: :meth:`logger.trace()<Logger.trace()>`
.. |logger.debug| replace:: :meth:`logger.debug()<Logger.debug()>`
.. |logger.info| replace:: :meth:`logger.info()<Logger.info()>`
.. |logger.success| replace:: :meth:`logger.success()<Logger.success()>`
.. |logger.warning| replace:: :meth:`logger.warning()<Logger.warning()>`
.. |logger.error| replace:: :meth:`logger.error()<Logger.error()>`
.. |logger.critical| replace:: :meth:`logger.critical()<Logger.critical()>`

.. |file-like object| replace:: ``file-like object``
.. _file-like object: https://docs.python.org/3/glossary.html#term-file-object
.. |callable| replace:: ``callable``
.. _callable: https://docs.python.org/3/library/functions.html#callable
.. |coroutine function| replace:: ``coroutine function``
.. _coroutine function: https://docs.python.org/3/glossary.html#term-coroutine-function
.. |re.Pattern| replace:: ``re.Pattern``
.. _re.Pattern: https://docs.python.org/3/library/re.html#re-objects
.. |multiprocessing.Context| replace:: ``multiprocessing.Context``
.. _multiprocessing.Context:
   https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

.. |better_exceptions| replace:: ``better_exceptions``
.. _better_exceptions: https://github.com/Qix-/better-exceptions

.. |loguru-config| replace:: ``loguru-config``
.. _loguru-config: https://github.com/erezinman/loguru-config

.. _Pendulum: https://pendulum.eustace.io/docs/#tokens

.. _@Qix-: https://github.com/Qix-
.. _@erezinman: https://github.com/erezinman
.. _@sdispater: https://github.com/sdispater

.. _formatting directives: https://docs.python.org/3/library/string.html#format-string-syntax
.. _reentrant: https://en.wikipedia.org/wiki/Reentrancy_(computing)
"""
import builtins
import contextlib
import functools
import logging
import re
import sys
import warnings
from collections import namedtuple
from inspect import isclass, iscoroutinefunction, isgeneratorfunction
from multiprocessing import current_process, get_context
from multiprocessing.context import BaseContext
from os.path import basename, splitext
from threading import current_thread
from . import _asyncio_loop, _colorama, _defaults, _filters
from ._better_exceptions import ExceptionFormatter
from ._colorizer import Colorizer
from ._contextvars import ContextVar
from ._datetime import aware_now
from ._error_interceptor import ErrorInterceptor
from ._file_sink import FileSink
from ._get_frame import get_frame
from ._handler import Handler
from ._locks_machinery import create_logger_lock
from ._recattrs import RecordException, RecordFile, RecordLevel, RecordProcess, RecordThread
from ._simple_sinks import AsyncSink, CallableSink, StandardSink, StreamSink
if sys.version_info >= (3, 6):
    from os import PathLike
else:
    from pathlib import PurePath as PathLike
Level = namedtuple('Level', ['name', 'no', 'color', 'icon'])
start_time = aware_now()
context = ContextVar('loguru_context', default={})

class Core:

    def __init__(self):
        if False:
            return 10
        levels = [Level('TRACE', _defaults.LOGURU_TRACE_NO, _defaults.LOGURU_TRACE_COLOR, _defaults.LOGURU_TRACE_ICON), Level('DEBUG', _defaults.LOGURU_DEBUG_NO, _defaults.LOGURU_DEBUG_COLOR, _defaults.LOGURU_DEBUG_ICON), Level('INFO', _defaults.LOGURU_INFO_NO, _defaults.LOGURU_INFO_COLOR, _defaults.LOGURU_INFO_ICON), Level('SUCCESS', _defaults.LOGURU_SUCCESS_NO, _defaults.LOGURU_SUCCESS_COLOR, _defaults.LOGURU_SUCCESS_ICON), Level('WARNING', _defaults.LOGURU_WARNING_NO, _defaults.LOGURU_WARNING_COLOR, _defaults.LOGURU_WARNING_ICON), Level('ERROR', _defaults.LOGURU_ERROR_NO, _defaults.LOGURU_ERROR_COLOR, _defaults.LOGURU_ERROR_ICON), Level('CRITICAL', _defaults.LOGURU_CRITICAL_NO, _defaults.LOGURU_CRITICAL_COLOR, _defaults.LOGURU_CRITICAL_ICON)]
        self.levels = {level.name: level for level in levels}
        self.levels_ansi_codes = {**{name: Colorizer.ansify(level.color) for (name, level) in self.levels.items()}, None: ''}
        self.levels_lookup = {name: (name, name, level.no, level.icon) for (name, level) in self.levels.items()}
        self.handlers_count = 0
        self.handlers = {}
        self.extra = {}
        self.patcher = None
        self.min_level = float('inf')
        self.enabled = {}
        self.activation_list = []
        self.activation_none = True
        self.lock = create_logger_lock()

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = self.__dict__.copy()
        state['lock'] = None
        return state

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        self.__dict__.update(state)
        self.lock = create_logger_lock()

class Logger:
    """An object to dispatch logging messages to configured handlers.

    The |Logger| is the core object of ``loguru``, every logging configuration and usage pass
    through a call to one of its methods. There is only one logger, so there is no need to retrieve
    one before usage.

    Once the ``logger`` is imported, it can be used to write messages about events happening in your
    code. By reading the output logs of your application, you gain a better understanding of the
    flow of your program and you more easily track and debug unexpected behaviors.

    Handlers to which the logger sends log messages are added using the |add| method. Note that you
    can use the |Logger| right after import as it comes pre-configured (logs are emitted to
    |sys.stderr| by default). Messages can be logged with different severity levels and they can be
    formatted using curly braces (it uses |str.format| under the hood).

    When a message is logged, a "record" is associated with it. This record is a dict which contains
    information about the logging context: time, function, file, line, thread, level... It also
    contains the ``__name__`` of the module, this is why you don't need named loggers.

    You should not instantiate a |Logger| by yourself, use ``from loguru import logger`` instead.
    """

    def __init__(self, core, exception, depth, record, lazy, colors, raw, capture, patchers, extra):
        if False:
            for i in range(10):
                print('nop')
        self._core = core
        self._options = (exception, depth, record, lazy, colors, raw, capture, patchers, extra)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<loguru.logger handlers=%r>' % list(self._core.handlers.values())

    def add(self, sink, *, level=_defaults.LOGURU_LEVEL, format=_defaults.LOGURU_FORMAT, filter=_defaults.LOGURU_FILTER, colorize=_defaults.LOGURU_COLORIZE, serialize=_defaults.LOGURU_SERIALIZE, backtrace=_defaults.LOGURU_BACKTRACE, diagnose=_defaults.LOGURU_DIAGNOSE, enqueue=_defaults.LOGURU_ENQUEUE, context=_defaults.LOGURU_CONTEXT, catch=_defaults.LOGURU_CATCH, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Add a handler sending log messages to a sink adequately configured.\n\n        Parameters\n        ----------\n        sink : |file-like object|_, |str|, |Path|, |callable|_, |coroutine function|_ or |Handler|\n            An object in charge of receiving formatted logging messages and propagating them to an\n            appropriate endpoint.\n        level : |int| or |str|, optional\n            The minimum severity level from which logged messages should be sent to the sink.\n        format : |str| or |callable|_, optional\n            The template used to format logged messages before being sent to the sink.\n        filter : |callable|_, |str| or |dict|, optional\n            A directive optionally used to decide for each logged message whether it should be sent\n            to the sink or not.\n        colorize : |bool|, optional\n            Whether the color markups contained in the formatted message should be converted to ansi\n            codes for terminal coloration, or stripped otherwise. If ``None``, the choice is\n            automatically made based on the sink being a tty or not.\n        serialize : |bool|, optional\n            Whether the logged message and its records should be first converted to a JSON string\n            before being sent to the sink.\n        backtrace : |bool|, optional\n            Whether the exception trace formatted should be extended upward, beyond the catching\n            point, to show the full stacktrace which generated the error.\n        diagnose : |bool|, optional\n            Whether the exception trace should display the variables values to eases the debugging.\n            This should be set to ``False`` in production to avoid leaking sensitive data.\n        enqueue : |bool|, optional\n            Whether the messages to be logged should first pass through a multiprocessing-safe queue\n            before reaching the sink. This is useful while logging to a file through multiple\n            processes. This also has the advantage of making logging calls non-blocking.\n        context : |multiprocessing.Context| or |str|, optional\n            A context object or name that will be used for all tasks involving internally the\n            |multiprocessing| module, in particular when ``enqueue=True``. If ``None``, the default\n            context is used.\n        catch : |bool|, optional\n            Whether errors occurring while sink handles logs messages should be automatically\n            caught. If ``True``, an exception message is displayed on |sys.stderr| but the exception\n            is not propagated to the caller, preventing your app to crash.\n        **kwargs\n            Additional parameters that are only valid to configure a coroutine or file sink (see\n            below).\n\n\n        If and only if the sink is a coroutine function, the following parameter applies:\n\n        Parameters\n        ----------\n        loop : |AbstractEventLoop|, optional\n            The event loop in which the asynchronous logging task will be scheduled and executed. If\n            ``None``, the loop used is the one returned by |asyncio.get_running_loop| at the time of\n            the logging call (task is discarded if there is no loop currently running).\n\n\n        If and only if the sink is a file path, the following parameters apply:\n\n        Parameters\n        ----------\n        rotation : |str|, |int|, |time|, |timedelta| or |callable|_, optional\n            A condition indicating whenever the current logged file should be closed and a new one\n            started.\n        retention : |str|, |int|, |timedelta| or |callable|_, optional\n            A directive filtering old files that should be removed during rotation or end of\n            program.\n        compression : |str| or |callable|_, optional\n            A compression or archive format to which log files should be converted at closure.\n        delay : |bool|, optional\n            Whether the file should be created as soon as the sink is configured, or delayed until\n            first logged message. It defaults to ``False``.\n        watch : |bool|, optional\n            Whether or not the file should be watched and re-opened when deleted or changed (based\n            on its device and inode properties) by an external program. It defaults to ``False``.\n        mode : |str|, optional\n            The opening mode as for built-in |open| function. It defaults to ``"a"`` (open the\n            file in appending mode).\n        buffering : |int|, optional\n            The buffering policy as for built-in |open| function. It defaults to ``1`` (line\n            buffered file).\n        encoding : |str|, optional\n            The file encoding as for built-in |open| function. It defaults to ``"utf8"``.\n        **kwargs\n            Others parameters are passed to the built-in |open| function.\n\n        Returns\n        -------\n        :class:`int`\n            An identifier associated with the added sink and which should be used to\n            |remove| it.\n\n        Raises\n        ------\n        ValueError\n            If any of the arguments passed to configure the sink is invalid.\n\n        Notes\n        -----\n        Extended summary follows.\n\n        .. _sink:\n\n        .. rubric:: The sink parameter\n\n        The ``sink`` handles incoming log messages and proceed to their writing somewhere and\n        somehow. A sink can take many forms:\n\n        - A |file-like object|_ like ``sys.stderr`` or ``open("file.log", "w")``. Anything with\n          a ``.write()`` method is considered as a file-like object. Custom handlers may also\n          implement ``flush()`` (called after each logged message), ``stop()`` (called at sink\n          termination) and ``complete()`` (awaited by the eponymous method).\n        - A file path as |str| or |Path|. It can be parametrized with some additional parameters,\n          see below.\n        - A |callable|_ (such as a simple function) like ``lambda msg: print(msg)``. This\n          allows for logging procedure entirely defined by user preferences and needs.\n        - A asynchronous |coroutine function|_ defined with the ``async def`` statement. The\n          coroutine object returned by such function will be added to the event loop using\n          |loop.create_task|. The tasks should be awaited before ending the loop by using\n          |complete|.\n        - A built-in |Handler| like ``logging.StreamHandler``. In such a case, the `Loguru` records\n          are automatically converted to the structure expected by the |logging| module.\n\n        Note that the logging functions are not `reentrant`_. This means you should avoid using\n        the ``logger`` inside any of your sinks or from within |signal| handlers. Otherwise, you\n        may face deadlock if the module\'s sink was not explicitly disabled.\n\n        .. _message:\n\n        .. rubric:: The logged message\n\n        The logged message passed to all added sinks is nothing more than a string of the\n        formatted log, to which a special attribute is associated: the ``.record`` which is a dict\n        containing all contextual information possibly needed (see below).\n\n        Logged messages are formatted according to the ``format`` of the added sink. This format\n        is usually a string containing braces fields to display attributes from the record dict.\n\n        If fine-grained control is needed, the ``format`` can also be a function which takes the\n        record as parameter and return the format template string. However, note that in such a\n        case, you should take care of appending the line ending and exception field to the returned\n        format, while ``"\\n{exception}"`` is automatically appended for convenience if ``format`` is\n        a string.\n\n        The ``filter`` attribute can be used to control which messages are effectively passed to the\n        sink and which one are ignored. A function can be used, accepting the record as an\n        argument, and returning ``True`` if the message should be logged, ``False`` otherwise. If\n        a string is used, only the records with the same ``name`` and its children will be allowed.\n        One can also pass a ``dict`` mapping module names to minimum required level. In such case,\n        each log record will search for it\'s closest parent in the ``dict`` and use the associated\n        level as the filter. The ``dict`` values can be ``int`` severity, ``str`` level name or\n        ``True`` and ``False`` to respectively authorize and discard all module logs\n        unconditionally. In order to set a default level, the ``""`` module name should be used as\n        it is the parent of all modules (it does not suppress global ``level`` threshold, though).\n\n        Note that while calling a logging method, the keyword arguments (if any) are automatically\n        added to the ``extra`` dict for convenient contextualization (in addition to being used for\n        formatting).\n\n        .. _levels:\n\n        .. rubric:: The severity levels\n\n        Each logged message is associated with a severity level. These levels make it possible to\n        prioritize messages and to choose the verbosity of the logs according to usages. For\n        example, it allows to display some debugging information to a developer, while hiding it to\n        the end user running the application.\n\n        The ``level`` attribute of every added sink controls the minimum threshold from which log\n        messages are allowed to be emitted. While using the ``logger``, you are in charge of\n        configuring the appropriate granularity of your logs. It is possible to add even more custom\n        levels by using the |level| method.\n\n        Here are the standard levels with their default severity value, each one is associated with\n        a logging method of the same name:\n\n        +----------------------+------------------------+------------------------+\n        | Level name           | Severity value         | Logger method          |\n        +======================+========================+========================+\n        | ``TRACE``            | 5                      | |logger.trace|         |\n        +----------------------+------------------------+------------------------+\n        | ``DEBUG``            | 10                     | |logger.debug|         |\n        +----------------------+------------------------+------------------------+\n        | ``INFO``             | 20                     | |logger.info|          |\n        +----------------------+------------------------+------------------------+\n        | ``SUCCESS``          | 25                     | |logger.success|       |\n        +----------------------+------------------------+------------------------+\n        | ``WARNING``          | 30                     | |logger.warning|       |\n        +----------------------+------------------------+------------------------+\n        | ``ERROR``            | 40                     | |logger.error|         |\n        +----------------------+------------------------+------------------------+\n        | ``CRITICAL``         | 50                     | |logger.critical|      |\n        +----------------------+------------------------+------------------------+\n\n        .. _record:\n\n        .. rubric:: The record dict\n\n        The record is just a Python dict, accessible from sinks by ``message.record``. It contains\n        all contextual information of the logging call (time, function, file, line, level, etc.).\n\n        Each of the record keys can be used in the handler\'s ``format`` so the corresponding value\n        is properly displayed in the logged message (e.g. ``"{level}"`` will return ``"INFO"``).\n        Some records\' values are objects with two or more attributes. These can be formatted with\n        ``"{key.attr}"`` (``"{key}"`` would display one by default).\n\n        Note that you can use any `formatting directives`_ available in Python\'s ``str.format()``\n        method (e.g. ``"{key: >3}"`` will right-align and pad to a width of 3 characters). This is\n        particularly useful for time formatting (see below).\n\n        +------------+---------------------------------+----------------------------+\n        | Key        | Description                     | Attributes                 |\n        +============+=================================+============================+\n        | elapsed    | The time elapsed since the      | See |timedelta|            |\n        |            | start of the program            |                            |\n        +------------+---------------------------------+----------------------------+\n        | exception  | The formatted exception if any, | ``type``, ``value``,       |\n        |            | ``None`` otherwise              | ``traceback``              |\n        +------------+---------------------------------+----------------------------+\n        | extra      | The dict of attributes          | None                       |\n        |            | bound by the user (see |bind|)  |                            |\n        +------------+---------------------------------+----------------------------+\n        | file       | The file where the logging call | ``name`` (default),        |\n        |            | was made                        | ``path``                   |\n        +------------+---------------------------------+----------------------------+\n        | function   | The function from which the     | None                       |\n        |            | logging call was made           |                            |\n        +------------+---------------------------------+----------------------------+\n        | level      | The severity used to log the    | ``name`` (default),        |\n        |            | message                         | ``no``, ``icon``           |\n        +------------+---------------------------------+----------------------------+\n        | line       | The line number in the source   | None                       |\n        |            | code                            |                            |\n        +------------+---------------------------------+----------------------------+\n        | message    | The logged message (not yet     | None                       |\n        |            | formatted)                      |                            |\n        +------------+---------------------------------+----------------------------+\n        | module     | The module where the logging    | None                       |\n        |            | call was made                   |                            |\n        +------------+---------------------------------+----------------------------+\n        | name       | The ``__name__`` where the      | None                       |\n        |            | logging call was made           |                            |\n        +------------+---------------------------------+----------------------------+\n        | process    | The process in which the        | ``name``, ``id`` (default) |\n        |            | logging call was made           |                            |\n        +------------+---------------------------------+----------------------------+\n        | thread     | The thread in which the         | ``name``, ``id`` (default) |\n        |            | logging call was made           |                            |\n        +------------+---------------------------------+----------------------------+\n        | time       | The aware local time when the   | See |datetime|             |\n        |            | logging call was made           |                            |\n        +------------+---------------------------------+----------------------------+\n\n        .. _time:\n\n        .. rubric:: The time formatting\n\n        To use your favorite time representation, you can set it directly in the time formatter\n        specifier of your handler format, like for example ``format="{time:HH:mm:ss} {message}"``.\n        Note that this datetime represents your local time, and it is also made timezone-aware,\n        so you can display the UTC offset to avoid ambiguities.\n\n        The time field can be formatted using more human-friendly tokens. These constitute a subset\n        of the one used by the `Pendulum`_ library of `@sdispater`_. To escape a token, just add\n        square brackets around it, for example ``"[YY]"`` would display literally ``"YY"``.\n\n        If you prefer to display UTC rather than local time, you can add ``"!UTC"`` at the very end\n        of the time format, like ``{time:HH:mm:ss!UTC}``. Doing so will convert the ``datetime``\n        to UTC before formatting.\n\n        If no time formatter specifier is used, like for example if ``format="{time} {message}"``,\n        the default one will use ISO 8601.\n\n        +------------------------+---------+----------------------------------------+\n        |                        | Token   | Output                                 |\n        +========================+=========+========================================+\n        | Year                   | YYYY    | 2000, 2001, 2002 ... 2012, 2013        |\n        |                        +---------+----------------------------------------+\n        |                        | YY      | 00, 01, 02 ... 12, 13                  |\n        +------------------------+---------+----------------------------------------+\n        | Quarter                | Q       | 1 2 3 4                                |\n        +------------------------+---------+----------------------------------------+\n        | Month                  | MMMM    | January, February, March ...           |\n        |                        +---------+----------------------------------------+\n        |                        | MMM     | Jan, Feb, Mar ...                      |\n        |                        +---------+----------------------------------------+\n        |                        | MM      | 01, 02, 03 ... 11, 12                  |\n        |                        +---------+----------------------------------------+\n        |                        | M       | 1, 2, 3 ... 11, 12                     |\n        +------------------------+---------+----------------------------------------+\n        | Day of Year            | DDDD    | 001, 002, 003 ... 364, 365             |\n        |                        +---------+----------------------------------------+\n        |                        | DDD     | 1, 2, 3 ... 364, 365                   |\n        +------------------------+---------+----------------------------------------+\n        | Day of Month           | DD      | 01, 02, 03 ... 30, 31                  |\n        |                        +---------+----------------------------------------+\n        |                        | D       | 1, 2, 3 ... 30, 31                     |\n        +------------------------+---------+----------------------------------------+\n        | Day of Week            | dddd    | Monday, Tuesday, Wednesday ...         |\n        |                        +---------+----------------------------------------+\n        |                        | ddd     | Mon, Tue, Wed ...                      |\n        |                        +---------+----------------------------------------+\n        |                        | d       | 0, 1, 2 ... 6                          |\n        +------------------------+---------+----------------------------------------+\n        | Days of ISO Week       | E       | 1, 2, 3 ... 7                          |\n        +------------------------+---------+----------------------------------------+\n        | Hour                   | HH      | 00, 01, 02 ... 23, 24                  |\n        |                        +---------+----------------------------------------+\n        |                        | H       | 0, 1, 2 ... 23, 24                     |\n        |                        +---------+----------------------------------------+\n        |                        | hh      | 01, 02, 03 ... 11, 12                  |\n        |                        +---------+----------------------------------------+\n        |                        | h       | 1, 2, 3 ... 11, 12                     |\n        +------------------------+---------+----------------------------------------+\n        | Minute                 | mm      | 00, 01, 02 ... 58, 59                  |\n        |                        +---------+----------------------------------------+\n        |                        | m       | 0, 1, 2 ... 58, 59                     |\n        +------------------------+---------+----------------------------------------+\n        | Second                 | ss      | 00, 01, 02 ... 58, 59                  |\n        |                        +---------+----------------------------------------+\n        |                        | s       | 0, 1, 2 ... 58, 59                     |\n        +------------------------+---------+----------------------------------------+\n        | Fractional Second      | S       | 0 1 ... 8 9                            |\n        |                        +---------+----------------------------------------+\n        |                        | SS      | 00, 01, 02 ... 98, 99                  |\n        |                        +---------+----------------------------------------+\n        |                        | SSS     | 000 001 ... 998 999                    |\n        |                        +---------+----------------------------------------+\n        |                        | SSSS... | 000[0..] 001[0..] ... 998[0..] 999[0..]|\n        |                        +---------+----------------------------------------+\n        |                        | SSSSSS  | 000000 000001 ... 999998 999999        |\n        +------------------------+---------+----------------------------------------+\n        | AM / PM                | A       | AM, PM                                 |\n        +------------------------+---------+----------------------------------------+\n        | Timezone               | Z       | -07:00, -06:00 ... +06:00, +07:00      |\n        |                        +---------+----------------------------------------+\n        |                        | ZZ      | -0700, -0600 ... +0600, +0700          |\n        |                        +---------+----------------------------------------+\n        |                        | zz      | EST CST ... MST PST                    |\n        +------------------------+---------+----------------------------------------+\n        | Seconds timestamp      | X       | 1381685817, 1234567890.123             |\n        +------------------------+---------+----------------------------------------+\n        | Microseconds timestamp | x       | 1234567890123                          |\n        +------------------------+---------+----------------------------------------+\n\n        .. _file:\n\n        .. rubric:: The file sinks\n\n        If the sink is a |str| or a |Path|, the corresponding file will be opened for writing logs.\n        The path can also contain a special ``"{time}"`` field that will be formatted with the\n        current date at file creation. The file is closed at sink stop, i.e. when the application\n        ends or the handler is removed.\n\n        The ``rotation`` check is made before logging each message. If there is already an existing\n        file with the same name that the file to be created, then the existing file is renamed by\n        appending the date to its basename to prevent file overwriting. This parameter accepts:\n\n        - an |int| which corresponds to the maximum file size in bytes before that the current\n          logged file is closed and a new one started over.\n        - a |timedelta| which indicates the frequency of each new rotation.\n        - a |time| which specifies the hour when the daily rotation should occur.\n        - a |str| for human-friendly parametrization of one of the previously enumerated types.\n          Examples: ``"100 MB"``, ``"0.5 GB"``, ``"1 month 2 weeks"``, ``"4 days"``, ``"10h"``,\n          ``"monthly"``, ``"18:00"``, ``"sunday"``, ``"w0"``, ``"monday at 12:00"``, ...\n        - a |callable|_ which will be invoked before logging. It should accept two arguments: the\n          logged message and the file object, and it should return ``True`` if the rotation should\n          happen now, ``False`` otherwise.\n\n        The ``retention`` occurs at rotation or at sink stop if rotation is ``None``. Files\n        resulting from previous sessions or rotations are automatically collected from disk. A file\n        is selected if it matches the pattern ``"basename(.*).ext(.*)"`` (possible time fields are\n        beforehand replaced with ``.*``) based on the configured sink. Afterwards, the list is\n        processed to determine files to be retained. This parameter accepts:\n\n        - an |int| which indicates the number of log files to keep, while older files are deleted.\n        - a |timedelta| which specifies the maximum age of files to keep.\n        - a |str| for human-friendly parametrization of the maximum age of files to keep.\n          Examples: ``"1 week, 3 days"``, ``"2 months"``, ...\n        - a |callable|_ which will be invoked before the retention process. It should accept the\n          list of log files as argument and process to whatever it wants (moving files, removing\n          them, etc.).\n\n        The ``compression`` happens at rotation or at sink stop if rotation is ``None``. This\n        parameter accepts:\n\n        - a |str| which corresponds to the compressed or archived file extension. This can be one\n          of: ``"gz"``, ``"bz2"``, ``"xz"``, ``"lzma"``, ``"tar"``, ``"tar.gz"``, ``"tar.bz2"``,\n          ``"tar.xz"``, ``"zip"``.\n        - a |callable|_ which will be invoked before file termination. It should accept the path of\n          the log file as argument and process to whatever it wants (custom compression, network\n          sending, removing it, etc.).\n\n        Either way, if you use a custom function designed according to your preferences, you must be\n        very careful not to use the ``logger`` within your function. Otherwise, there is a risk that\n        your program hang because of a deadlock.\n\n        .. _color:\n\n        .. rubric:: The color markups\n\n        To add colors to your logs, you just have to enclose your format string with the appropriate\n        tags (e.g. ``<red>some message</red>``). These tags are automatically removed if the sink\n        doesn\'t support ansi codes. For convenience, you can use ``</>`` to close the last opening\n        tag without repeating its name (e.g. ``<red>another message</>``).\n\n        The special tag ``<level>`` (abbreviated with ``<lvl>``) is transformed according to\n        the configured color of the logged message level.\n\n        Tags which are not recognized will raise an exception during parsing, to inform you about\n        possible misuse. If you wish to display a markup tag literally, you can escape it by\n        prepending a ``\\`` like for example ``\\<blue>``. If, for some reason, you need to escape a\n        string programmatically, note that the regex used internally to parse markup tags is\n        ``r"\\\\?</?((?:[fb]g\\s)?[^<>\\s]*)>"``.\n\n        Note that when logging a message with ``opt(colors=True)``, color tags present in the\n        formatting arguments (``args`` and ``kwargs``) are completely ignored. This is important if\n        you need to log strings containing markups that might interfere with the color tags (in this\n        case, do not use f-string).\n\n        Here are the available tags (note that compatibility may vary depending on terminal):\n\n        +------------------------------------+--------------------------------------+\n        | Color (abbr)                       | Styles (abbr)                        |\n        +====================================+======================================+\n        | Black (k)                          | Bold (b)                             |\n        +------------------------------------+--------------------------------------+\n        | Blue (e)                           | Dim (d)                              |\n        +------------------------------------+--------------------------------------+\n        | Cyan (c)                           | Normal (n)                           |\n        +------------------------------------+--------------------------------------+\n        | Green (g)                          | Italic (i)                           |\n        +------------------------------------+--------------------------------------+\n        | Magenta (m)                        | Underline (u)                        |\n        +------------------------------------+--------------------------------------+\n        | Red (r)                            | Strike (s)                           |\n        +------------------------------------+--------------------------------------+\n        | White (w)                          | Reverse (v)                          |\n        +------------------------------------+--------------------------------------+\n        | Yellow (y)                         | Blink (l)                            |\n        +------------------------------------+--------------------------------------+\n        |                                    | Hide (h)                             |\n        +------------------------------------+--------------------------------------+\n\n        Usage:\n\n        +-----------------+-------------------------------------------------------------------+\n        | Description     | Examples                                                          |\n        |                 +---------------------------------+---------------------------------+\n        |                 | Foreground                      | Background                      |\n        +=================+=================================+=================================+\n        | Basic colors    | ``<red>``, ``<r>``              | ``<GREEN>``, ``<G>``            |\n        +-----------------+---------------------------------+---------------------------------+\n        | Light colors    | ``<light-blue>``, ``<le>``      | ``<LIGHT-CYAN>``, ``<LC>``      |\n        +-----------------+---------------------------------+---------------------------------+\n        | 8-bit colors    | ``<fg 86>``, ``<fg 255>``       | ``<bg 42>``, ``<bg 9>``         |\n        +-----------------+---------------------------------+---------------------------------+\n        | Hex colors      | ``<fg #00005f>``, ``<fg #EE1>`` | ``<bg #AF5FD7>``, ``<bg #fff>`` |\n        +-----------------+---------------------------------+---------------------------------+\n        | RGB colors      | ``<fg 0,95,0>``                 | ``<bg 72,119,65>``              |\n        +-----------------+---------------------------------+---------------------------------+\n        | Stylizing       | ``<bold>``, ``<b>``,  ``<underline>``, ``<u>``                    |\n        +-----------------+-------------------------------------------------------------------+\n\n        .. _env:\n\n        .. rubric:: The environment variables\n\n        The default values of sink parameters can be entirely customized. This is particularly\n        useful if you don\'t like the log format of the pre-configured sink.\n\n        Each of the |add| default parameter can be modified by setting the ``LOGURU_[PARAM]``\n        environment variable. For example on Linux: ``export LOGURU_FORMAT="{time} - {message}"``\n        or ``export LOGURU_DIAGNOSE=NO``.\n\n        The default levels\' attributes can also be modified by setting the ``LOGURU_[LEVEL]_[ATTR]``\n        environment variable. For example, on Windows: ``setx LOGURU_DEBUG_COLOR "<blue>"``\n        or ``setx LOGURU_TRACE_ICON "🚀"``. If you use the ``set`` command, do not include quotes\n        but escape special symbol as needed, e.g. ``set LOGURU_DEBUG_COLOR=^<blue^>``.\n\n        If you want to disable the pre-configured sink, you can set the ``LOGURU_AUTOINIT``\n        variable to ``False``.\n\n        On Linux, you will probably need to edit the ``~/.profile`` file to make this persistent. On\n        Windows, don\'t forget to restart your terminal for the change to be taken into account.\n\n        Examples\n        --------\n        >>> logger.add(sys.stdout, format="{time} - {level} - {message}", filter="sub.module")\n\n        >>> logger.add("file_{time}.log", level="TRACE", rotation="100 MB")\n\n        >>> def debug_only(record):\n        ...     return record["level"].name == "DEBUG"\n        ...\n        >>> logger.add("debug.log", filter=debug_only)  # Other levels are filtered out\n\n        >>> def my_sink(message):\n        ...     record = message.record\n        ...     update_db(message, time=record["time"], level=record["level"])\n        ...\n        >>> logger.add(my_sink)\n\n        >>> level_per_module = {\n        ...     "": "DEBUG",\n        ...     "third.lib": "WARNING",\n        ...     "anotherlib": False\n        ... }\n        >>> logger.add(lambda m: print(m, end=""), filter=level_per_module, level=0)\n\n        >>> async def publish(message):\n        ...     await api.post(message)\n        ...\n        >>> logger.add(publish, serialize=True)\n\n        >>> from logging import StreamHandler\n        >>> logger.add(StreamHandler(sys.stderr), format="{message}")\n\n        >>> class RandomStream:\n        ...     def __init__(self, seed, threshold):\n        ...         self.threshold = threshold\n        ...         random.seed(seed)\n        ...     def write(self, message):\n        ...         if random.random() > self.threshold:\n        ...             print(message)\n        ...\n        >>> stream_object = RandomStream(seed=12345, threshold=0.25)\n        >>> logger.add(stream_object, level="INFO")\n        '
        with self._core.lock:
            handler_id = self._core.handlers_count
            self._core.handlers_count += 1
        error_interceptor = ErrorInterceptor(catch, handler_id)
        if colorize is None and serialize:
            colorize = False
        if isinstance(sink, (str, PathLike)):
            path = sink
            name = "'%s'" % path
            if colorize is None:
                colorize = False
            wrapped_sink = FileSink(path, **kwargs)
            kwargs = {}
            encoding = wrapped_sink.encoding
            terminator = '\n'
            exception_prefix = ''
        elif hasattr(sink, 'write') and callable(sink.write):
            name = getattr(sink, 'name', None) or repr(sink)
            if colorize is None:
                colorize = _colorama.should_colorize(sink)
            if colorize is True and _colorama.should_wrap(sink):
                stream = _colorama.wrap(sink)
            else:
                stream = sink
            wrapped_sink = StreamSink(stream)
            encoding = getattr(sink, 'encoding', None)
            terminator = '\n'
            exception_prefix = ''
        elif isinstance(sink, logging.Handler):
            name = repr(sink)
            if colorize is None:
                colorize = False
            wrapped_sink = StandardSink(sink)
            encoding = getattr(sink, 'encoding', None)
            terminator = ''
            exception_prefix = '\n'
        elif iscoroutinefunction(sink) or iscoroutinefunction(getattr(sink, '__call__', None)):
            name = getattr(sink, '__name__', None) or repr(sink)
            if colorize is None:
                colorize = False
            loop = kwargs.pop('loop', None)
            if enqueue and loop is None:
                try:
                    loop = _asyncio_loop.get_running_loop()
                except RuntimeError as e:
                    raise ValueError('An event loop is required to add a coroutine sink with `enqueue=True`, but none has been passed as argument and none is currently running.') from e
            coro = sink if iscoroutinefunction(sink) else sink.__call__
            wrapped_sink = AsyncSink(coro, loop, error_interceptor)
            encoding = 'utf8'
            terminator = '\n'
            exception_prefix = ''
        elif callable(sink):
            name = getattr(sink, '__name__', None) or repr(sink)
            if colorize is None:
                colorize = False
            wrapped_sink = CallableSink(sink)
            encoding = 'utf8'
            terminator = '\n'
            exception_prefix = ''
        else:
            raise TypeError("Cannot log to objects of type '%s'" % type(sink).__name__)
        if kwargs:
            raise TypeError("add() got an unexpected keyword argument '%s'" % next(iter(kwargs)))
        if filter is None:
            filter_func = None
        elif filter == '':
            filter_func = _filters.filter_none
        elif isinstance(filter, str):
            parent = filter + '.'
            length = len(parent)
            filter_func = functools.partial(_filters.filter_by_name, parent=parent, length=length)
        elif isinstance(filter, dict):
            level_per_module = {}
            for (module, level_) in filter.items():
                if module is not None and (not isinstance(module, str)):
                    raise TypeError("The filter dict contains an invalid module, it should be a string (or None), not: '%s'" % type(module).__name__)
                if level_ is False:
                    levelno_ = False
                elif level_ is True:
                    levelno_ = 0
                elif isinstance(level_, str):
                    try:
                        levelno_ = self.level(level_).no
                    except ValueError:
                        raise ValueError("The filter dict contains a module '%s' associated to a level name which does not exist: '%s'" % (module, level_)) from None
                elif isinstance(level_, int):
                    levelno_ = level_
                else:
                    raise TypeError("The filter dict contains a module '%s' associated to an invalid level, it should be an integer, a string or a boolean, not: '%s'" % (module, type(level_).__name__))
                if levelno_ < 0:
                    raise ValueError("The filter dict contains a module '%s' associated to an invalid level, it should be a positive integer, not: '%d'" % (module, levelno_))
                level_per_module[module] = levelno_
            filter_func = functools.partial(_filters.filter_by_level, level_per_module=level_per_module)
        elif callable(filter):
            if filter == builtins.filter:
                raise ValueError("The built-in 'filter()' function cannot be used as a 'filter' parameter, this is most likely a mistake (please double-check the arguments passed to 'logger.add()').")
            filter_func = filter
        else:
            raise TypeError("Invalid filter, it should be a function, a string or a dict, not: '%s'" % type(filter).__name__)
        if isinstance(level, str):
            levelno = self.level(level).no
        elif isinstance(level, int):
            levelno = level
        else:
            raise TypeError("Invalid level, it should be an integer or a string, not: '%s'" % type(level).__name__)
        if levelno < 0:
            raise ValueError('Invalid level value, it should be a positive integer, not: %d' % levelno)
        if isinstance(format, str):
            try:
                formatter = Colorizer.prepare_format(format + terminator + '{exception}')
            except ValueError as e:
                raise ValueError('Invalid format, color markups could not be parsed correctly') from e
            is_formatter_dynamic = False
        elif callable(format):
            if format == builtins.format:
                raise ValueError("The built-in 'format()' function cannot be used as a 'format' parameter, this is most likely a mistake (please double-check the arguments passed to 'logger.add()').")
            formatter = format
            is_formatter_dynamic = True
        else:
            raise TypeError("Invalid format, it should be a string or a function, not: '%s'" % type(format).__name__)
        if not isinstance(encoding, str):
            encoding = 'ascii'
        if isinstance(context, str):
            context = get_context(context)
        elif context is not None and (not isinstance(context, BaseContext)):
            raise TypeError("Invalid context, it should be a string or a multiprocessing context, not: '%s'" % type(context).__name__)
        with self._core.lock:
            exception_formatter = ExceptionFormatter(colorize=colorize, encoding=encoding, diagnose=diagnose, backtrace=backtrace, hidden_frames_filename=self.catch.__code__.co_filename, prefix=exception_prefix)
            handler = Handler(name=name, sink=wrapped_sink, levelno=levelno, formatter=formatter, is_formatter_dynamic=is_formatter_dynamic, filter_=filter_func, colorize=colorize, serialize=serialize, enqueue=enqueue, multiprocessing_context=context, id_=handler_id, error_interceptor=error_interceptor, exception_formatter=exception_formatter, levels_ansi_codes=self._core.levels_ansi_codes)
            handlers = self._core.handlers.copy()
            handlers[handler_id] = handler
            self._core.min_level = min(self._core.min_level, levelno)
            self._core.handlers = handlers
        return handler_id

    def remove(self, handler_id=None):
        if False:
            while True:
                i = 10
        'Remove a previously added handler and stop sending logs to its sink.\n\n        Parameters\n        ----------\n        handler_id : |int| or ``None``\n            The id of the sink to remove, as it was returned by the |add| method. If ``None``, all\n            handlers are removed. The pre-configured handler is guaranteed to have the index ``0``.\n\n        Raises\n        ------\n        ValueError\n            If ``handler_id`` is not ``None`` but there is no active handler with such id.\n\n        Examples\n        --------\n        >>> i = logger.add(sys.stderr, format="{message}")\n        >>> logger.info("Logging")\n        Logging\n        >>> logger.remove(i)\n        >>> logger.info("No longer logging")\n        '
        if not (handler_id is None or isinstance(handler_id, int)):
            raise TypeError("Invalid handler id, it should be an integer as returned by the 'add()' method (or None), not: '%s'" % type(handler_id).__name__)
        with self._core.lock:
            handlers = self._core.handlers.copy()
            if handler_id is not None and handler_id not in handlers:
                raise ValueError('There is no existing handler with id %d' % handler_id) from None
            if handler_id is None:
                handler_ids = list(handlers.keys())
            else:
                handler_ids = [handler_id]
            for handler_id in handler_ids:
                handler = handlers.pop(handler_id)
                levelnos = (h.levelno for h in handlers.values())
                self._core.min_level = min(levelnos, default=float('inf'))
                self._core.handlers = handlers
                handler.stop()

    def complete(self):
        if False:
            while True:
                i = 10
        'Wait for the end of enqueued messages and asynchronous tasks scheduled by handlers.\n\n        This method proceeds in two steps: first it waits for all logging messages added to handlers\n        with ``enqueue=True`` to be processed, then it returns an object that can be awaited to\n        finalize all logging tasks added to the event loop by coroutine sinks.\n\n        It can be called from non-asynchronous code. This is especially recommended when the\n        ``logger`` is utilized with ``multiprocessing`` to ensure messages put to the internal\n        queue have been properly transmitted before leaving a child process.\n\n        The returned object should be awaited before the end of a coroutine executed by\n        |asyncio.run| or |loop.run_until_complete| to ensure all asynchronous logging messages are\n        processed. The function |asyncio.get_running_loop| is called beforehand, only tasks\n        scheduled in the same loop that the current one will be awaited by the method.\n\n        Returns\n        -------\n        :term:`awaitable`\n            An awaitable object which ensures all asynchronous logging calls are completed when\n            awaited.\n\n        Examples\n        --------\n        >>> async def sink(message):\n        ...     await asyncio.sleep(0.1)  # IO processing...\n        ...     print(message, end="")\n        ...\n        >>> async def work():\n        ...     logger.info("Start")\n        ...     logger.info("End")\n        ...     await logger.complete()\n        ...\n        >>> logger.add(sink)\n        1\n        >>> asyncio.run(work())\n        Start\n        End\n\n        >>> def process():\n        ...     logger.info("Message sent from the child")\n        ...     logger.complete()\n        ...\n        >>> logger.add(sys.stderr, enqueue=True)\n        1\n        >>> process = multiprocessing.Process(target=process)\n        >>> process.start()\n        >>> process.join()\n        Message sent from the child\n        '
        tasks = []
        with self._core.lock:
            handlers = self._core.handlers.copy()
            for handler in handlers.values():
                handler.complete_queue()
                tasks.extend(handler.tasks_to_complete())

        class AwaitableCompleter:

            def __await__(self):
                if False:
                    i = 10
                    return i + 15
                for task in tasks:
                    yield from task.__await__()
        return AwaitableCompleter()

    def catch(self, exception=Exception, *, level='ERROR', reraise=False, onerror=None, exclude=None, default=None, message="An error has been caught in function '{record[function]}', process '{record[process].name}' ({record[process].id}), thread '{record[thread].name}' ({record[thread].id}):"):
        if False:
            for i in range(10):
                print('nop')
        'Return a decorator to automatically log possibly caught error in wrapped function.\n\n        This is useful to ensure unexpected exceptions are logged, the entire program can be\n        wrapped by this method. This is also very useful to decorate |Thread.run| methods while\n        using threads to propagate errors to the main logger thread.\n\n        Note that the visibility of variables values (which uses the great |better_exceptions|_\n        library from `@Qix-`_) depends on the ``diagnose`` option of each configured sink.\n\n        The returned object can also be used as a context manager.\n\n        Parameters\n        ----------\n        exception : |Exception|, optional\n            The type of exception to intercept. If several types should be caught, a tuple of\n            exceptions can be used too.\n        level : |str| or |int|, optional\n            The level name or severity with which the message should be logged.\n        reraise : |bool|, optional\n            Whether the exception should be raised again and hence propagated to the caller.\n        onerror : |callable|_, optional\n            A function that will be called if an error occurs, once the message has been logged.\n            It should accept the exception instance as it sole argument.\n        exclude : |Exception|, optional\n            A type of exception (or a tuple of types) that will be purposely ignored and hence\n            propagated to the caller without being logged.\n        default : |Any|, optional\n            The value to be returned by the decorated function if an error occurred without being\n            re-raised.\n        message : |str|, optional\n            The message that will be automatically logged if an exception occurs. Note that it will\n            be formatted with the ``record`` attribute.\n\n        Returns\n        -------\n        :term:`decorator` / :term:`context manager`\n            An object that can be used to decorate a function or as a context manager to log\n            exceptions possibly caught.\n\n        Examples\n        --------\n        >>> @logger.catch\n        ... def f(x):\n        ...     100 / x\n        ...\n        >>> def g():\n        ...     f(10)\n        ...     f(0)\n        ...\n        >>> g()\n        ERROR - An error has been caught in function \'g\', process \'Main\' (367), thread \'ch1\' (1398):\n        Traceback (most recent call last):\n          File "program.py", line 12, in <module>\n            g()\n            └ <function g at 0x7f225fe2bc80>\n        > File "program.py", line 10, in g\n            f(0)\n            └ <function f at 0x7f225fe2b9d8>\n          File "program.py", line 6, in f\n            100 / x\n                  └ 0\n        ZeroDivisionError: division by zero\n\n        >>> with logger.catch(message="Because we never know..."):\n        ...    main()  # No exception, no logs\n\n        >>> # Use \'onerror\' to prevent the program exit code to be 0 (if \'reraise=False\') while\n        >>> # also avoiding the stacktrace to be duplicated on stderr (if \'reraise=True\').\n        >>> @logger.catch(onerror=lambda _: sys.exit(1))\n        ... def main():\n        ...     1 / 0\n        '
        if callable(exception) and (not isclass(exception) or not issubclass(exception, BaseException)):
            return self.catch()(exception)
        logger = self

        class Catcher:

            def __init__(self, from_decorator):
                if False:
                    while True:
                        i = 10
                self._from_decorator = from_decorator

            def __enter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return None

            def __exit__(self, type_, value, traceback_):
                if False:
                    print('Hello World!')
                if type_ is None:
                    return
                if not issubclass(type_, exception):
                    return False
                if exclude is not None and issubclass(type_, exclude):
                    return False
                from_decorator = self._from_decorator
                (_, depth, _, *options) = logger._options
                if from_decorator:
                    depth += 1
                catch_options = [(type_, value, traceback_), depth, True] + options
                logger._log(level, from_decorator, catch_options, message, (), {})
                if onerror is not None:
                    onerror(value)
                return not reraise

            def __call__(self, function):
                if False:
                    while True:
                        i = 10
                if isclass(function):
                    raise TypeError("Invalid object decorated with 'catch()', it must be a function, not a class (tried to wrap '%s')" % function.__name__)
                catcher = Catcher(True)
                if iscoroutinefunction(function):

                    async def catch_wrapper(*args, **kwargs):
                        with catcher:
                            return await function(*args, **kwargs)
                        return default
                elif isgeneratorfunction(function):

                    def catch_wrapper(*args, **kwargs):
                        if False:
                            for i in range(10):
                                print('nop')
                        with catcher:
                            return (yield from function(*args, **kwargs))
                        return default
                else:

                    def catch_wrapper(*args, **kwargs):
                        if False:
                            while True:
                                i = 10
                        with catcher:
                            return function(*args, **kwargs)
                        return default
                functools.update_wrapper(catch_wrapper, function)
                return catch_wrapper
        return Catcher(False)

    def opt(self, *, exception=None, record=False, lazy=False, colors=False, raw=False, capture=True, depth=0, ansi=False):
        if False:
            for i in range(10):
                print('nop')
        'Parametrize a logging call to slightly change generated log message.\n\n        Note that it\'s not possible to chain |opt| calls, the last one takes precedence over the\n        others as it will "reset" the options to their default values.\n\n        Parameters\n        ----------\n        exception : |bool|, |tuple| or |Exception|, optional\n            If it does not evaluate as ``False``, the passed exception is formatted and added to the\n            log message. It could be an |Exception| object or a ``(type, value, traceback)`` tuple,\n            otherwise the exception information is retrieved from |sys.exc_info|.\n        record : |bool|, optional\n            If ``True``, the record dict contextualizing the logging call can be used to format the\n            message by using ``{record[key]}`` in the log message.\n        lazy : |bool|, optional\n            If ``True``, the logging call attribute to format the message should be functions which\n            will be called only if the level is high enough. This can be used to avoid expensive\n            functions if not necessary.\n        colors : |bool|, optional\n            If ``True``, logged message will be colorized according to the markups it possibly\n            contains.\n        raw : |bool|, optional\n            If ``True``, the formatting of each sink will be bypassed and the message will be sent\n            as is.\n        capture : |bool|, optional\n            If ``False``, the ``**kwargs`` of logged message will not automatically populate\n            the ``extra`` dict (although they are still used for formatting).\n        depth : |int|, optional\n            Specify which stacktrace should be used to contextualize the logged message. This is\n            useful while using the logger from inside a wrapped function to retrieve worthwhile\n            information.\n        ansi : |bool|, optional\n            Deprecated since version 0.4.1: the ``ansi`` parameter will be removed in Loguru 1.0.0,\n            it is replaced by ``colors`` which is a more appropriate name.\n\n        Returns\n        -------\n        :class:`~Logger`\n            A logger wrapping the core logger, but transforming logged message adequately before\n            sending.\n\n        Examples\n        --------\n        >>> try:\n        ...     1 / 0\n        ... except ZeroDivisionError:\n        ...    logger.opt(exception=True).debug("Exception logged with debug level:")\n        ...\n        [18:10:02] DEBUG in \'<module>\' - Exception logged with debug level:\n        Traceback (most recent call last, catch point marked):\n        > File "<stdin>", line 2, in <module>\n        ZeroDivisionError: division by zero\n\n        >>> logger.opt(record=True).info("Current line is: {record[line]}")\n        [18:10:33] INFO in \'<module>\' - Current line is: 1\n\n        >>> logger.opt(lazy=True).debug("If sink <= DEBUG: {x}", x=lambda: math.factorial(2**5))\n        [18:11:19] DEBUG in \'<module>\' - If sink <= DEBUG: 263130836933693530167218012160000000\n\n        >>> logger.opt(colors=True).warning("We got a <red>BIG</red> problem")\n        [18:11:30] WARNING in \'<module>\' - We got a BIG problem\n\n        >>> logger.opt(raw=True).debug("No formatting\\n")\n        No formatting\n\n        >>> logger.opt(capture=False).info("Displayed but not captured: {value}", value=123)\n        [18:11:41] Displayed but not captured: 123\n\n        >>> def wrapped():\n        ...     logger.opt(depth=1).info("Get parent context")\n        ...\n        >>> def func():\n        ...     wrapped()\n        ...\n        >>> func()\n        [18:11:54] DEBUG in \'func\' - Get parent context\n        '
        if ansi:
            colors = True
            warnings.warn("The 'ansi' parameter is deprecated, please use 'colors' instead", DeprecationWarning, stacklevel=2)
        args = self._options[-2:]
        return Logger(self._core, exception, depth, record, lazy, colors, raw, capture, *args)

    def bind(__self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Bind attributes to the ``extra`` dict of each logged message record.\n\n        This is used to add custom context to each logging call.\n\n        Parameters\n        ----------\n        **kwargs\n            Mapping between keys and values that will be added to the ``extra`` dict.\n\n        Returns\n        -------\n        :class:`~Logger`\n            A logger wrapping the core logger, but which sends record with the customized ``extra``\n            dict.\n\n        Examples\n        --------\n        >>> logger.add(sys.stderr, format="{extra[ip]} - {message}")\n        >>> class Server:\n        ...     def __init__(self, ip):\n        ...         self.ip = ip\n        ...         self.logger = logger.bind(ip=ip)\n        ...     def call(self, message):\n        ...         self.logger.info(message)\n        ...\n        >>> instance_1 = Server("192.168.0.200")\n        >>> instance_2 = Server("127.0.0.1")\n        >>> instance_1.call("First instance")\n        192.168.0.200 - First instance\n        >>> instance_2.call("Second instance")\n        127.0.0.1 - Second instance\n        '
        (*options, extra) = __self._options
        return Logger(__self._core, *options, {**extra, **kwargs})

    @contextlib.contextmanager
    def contextualize(__self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Bind attributes to the context-local ``extra`` dict while inside the ``with`` block.\n\n        Contrary to |bind| there is no ``logger`` returned, the ``extra`` dict is modified in-place\n        and updated globally. Most importantly, it uses |contextvars| which means that\n        contextualized values are unique to each threads and asynchronous tasks.\n\n        The ``extra`` dict will retrieve its initial state once the context manager is exited.\n\n        Parameters\n        ----------\n        **kwargs\n            Mapping between keys and values that will be added to the context-local ``extra`` dict.\n\n        Returns\n        -------\n        :term:`context manager` / :term:`decorator`\n            A context manager (usable as a decorator too) that will bind the attributes once entered\n            and restore the initial state of the ``extra`` dict while exited.\n\n        Examples\n        --------\n        >>> logger.add(sys.stderr, format="{message} | {extra}")\n        1\n        >>> def task():\n        ...     logger.info("Processing!")\n        ...\n        >>> with logger.contextualize(task_id=123):\n        ...     task()\n        ...\n        Processing! | {\'task_id\': 123}\n        >>> logger.info("Done.")\n        Done. | {}\n        '
        with __self._core.lock:
            new_context = {**context.get(), **kwargs}
            token = context.set(new_context)
        try:
            yield
        finally:
            with __self._core.lock:
                context.reset(token)

    def patch(self, patcher):
        if False:
            for i in range(10):
                print('nop')
        'Attach a function to modify the record dict created by each logging call.\n\n        The ``patcher`` may be used to update the record on-the-fly before it\'s propagated to the\n        handlers. This allows the "extra" dict to be populated with dynamic values and also permits\n        advanced modifications of the record emitted while logging a message. The function is called\n        once before sending the log message to the different handlers.\n\n        It is recommended to apply modification on the ``record["extra"]`` dict rather than on the\n        ``record`` dict itself, as some values are used internally by `Loguru`, and modify them may\n        produce unexpected results.\n\n        The logger can be patched multiple times. In this case, the functions are called in the\n        same order as they are added.\n\n        Parameters\n        ----------\n        patcher: |callable|_\n            The function to which the record dict will be passed as the sole argument. This function\n            is in charge of updating the record in-place, the function does not need to return any\n            value, the modified record object will be re-used.\n\n        Returns\n        -------\n        :class:`~Logger`\n            A logger wrapping the core logger, but which records are passed through the ``patcher``\n            function before being sent to the added handlers.\n\n        Examples\n        --------\n        >>> logger.add(sys.stderr, format="{extra[utc]} {message}")\n        >>> logger = logger.patch(lambda record: record["extra"].update(utc=datetime.utcnow())\n        >>> logger.info("That\'s way, you can log messages with time displayed in UTC")\n\n        >>> def wrapper(func):\n        ...     @functools.wraps(func)\n        ...     def wrapped(*args, **kwargs):\n        ...         logger.patch(lambda r: r.update(function=func.__name__)).info("Wrapped!")\n        ...         return func(*args, **kwargs)\n        ...     return wrapped\n\n        >>> def recv_record_from_network(pipe):\n        ...     record = pickle.loads(pipe.read())\n        ...     level, message = record["level"], record["message"]\n        ...     logger.patch(lambda r: r.update(record)).log(level, message)\n        '
        (*options, patchers, extra) = self._options
        return Logger(self._core, *options, patchers + [patcher], extra)

    def level(self, name, no=None, color=None, icon=None):
        if False:
            for i in range(10):
                print('nop')
        'Add, update or retrieve a logging level.\n\n        Logging levels are defined by their ``name`` to which a severity ``no``, an ansi ``color``\n        tag and an ``icon`` are associated and possibly modified at run-time. To |log| to a custom\n        level, you should necessarily use its name, the severity number is not linked back to levels\n        name (this implies that several levels can share the same severity).\n\n        To add a new level, its ``name`` and its ``no`` are required. A ``color`` and an ``icon``\n        can also be specified or will be empty by default.\n\n        To update an existing level, pass its ``name`` with the parameters to be changed. It is not\n        possible to modify the ``no`` of a level once it has been added.\n\n        To retrieve level information, the ``name`` solely suffices.\n\n        Parameters\n        ----------\n        name : |str|\n            The name of the logging level.\n        no : |int|\n            The severity of the level to be added or updated.\n        color : |str|\n            The color markup of the level to be added or updated.\n        icon : |str|\n            The icon of the level to be added or updated.\n\n        Returns\n        -------\n        ``Level``\n            A |namedtuple| containing information about the level.\n\n        Raises\n        ------\n        ValueError\n            If there is no level registered with such ``name``.\n\n        Examples\n        --------\n        >>> level = logger.level("ERROR")\n        >>> print(level)\n        Level(name=\'ERROR\', no=40, color=\'<red><bold>\', icon=\'❌\')\n        >>> logger.add(sys.stderr, format="{level.no} {level.icon} {message}")\n        1\n        >>> logger.level("CUSTOM", no=15, color="<blue>", icon="@")\n        Level(name=\'CUSTOM\', no=15, color=\'<blue>\', icon=\'@\')\n        >>> logger.log("CUSTOM", "Logging...")\n        15 @ Logging...\n        >>> logger.level("WARNING", icon=r"/!\\")\n        Level(name=\'WARNING\', no=30, color=\'<yellow><bold>\', icon=\'/!\\\\\')\n        >>> logger.warning("Updated!")\n        30 /!\\ Updated!\n        '
        if not isinstance(name, str):
            raise TypeError("Invalid level name, it should be a string, not: '%s'" % type(name).__name__)
        if no is color is icon is None:
            try:
                return self._core.levels[name]
            except KeyError:
                raise ValueError("Level '%s' does not exist" % name) from None
        if name not in self._core.levels:
            if no is None:
                raise ValueError("Level '%s' does not exist, you have to create it by specifying a level no" % name)
            else:
                (old_color, old_icon) = ('', ' ')
        elif no is not None:
            raise TypeError("Level '%s' already exists, you can't update its severity no" % name)
        else:
            (_, no, old_color, old_icon) = self.level(name)
        if color is None:
            color = old_color
        if icon is None:
            icon = old_icon
        if not isinstance(no, int):
            raise TypeError("Invalid level no, it should be an integer, not: '%s'" % type(no).__name__)
        if no < 0:
            raise ValueError('Invalid level no, it should be a positive integer, not: %d' % no)
        ansi = Colorizer.ansify(color)
        level = Level(name, no, color, icon)
        with self._core.lock:
            self._core.levels[name] = level
            self._core.levels_ansi_codes[name] = ansi
            self._core.levels_lookup[name] = (name, name, no, icon)
            for handler in self._core.handlers.values():
                handler.update_format(name)
        return level

    def disable(self, name):
        if False:
            return 10
        'Disable logging of messages coming from ``name`` module and its children.\n\n        Developers of library using `Loguru` should absolutely disable it to avoid disrupting\n        users with unrelated logs messages.\n\n        Note that in some rare circumstances, it is not possible for `Loguru` to\n        determine the module\'s ``__name__`` value. In such situation, ``record["name"]`` will be\n        equal to ``None``, this is why ``None`` is also a valid argument.\n\n        Parameters\n        ----------\n        name : |str| or ``None``\n            The name of the parent module to disable.\n\n        Examples\n        --------\n        >>> logger.info("Allowed message by default")\n        [22:21:55] Allowed message by default\n        >>> logger.disable("my_library")\n        >>> logger.info("While publishing a library, don\'t forget to disable logging")\n        '
        self._change_activation(name, False)

    def enable(self, name):
        if False:
            return 10
        'Enable logging of messages coming from ``name`` module and its children.\n\n        Logging is generally disabled by imported library using `Loguru`, hence this function\n        allows users to receive these messages anyway.\n\n        To enable all logs regardless of the module they are coming from, an empty string ``""`` can\n        be passed.\n\n        Parameters\n        ----------\n        name : |str| or ``None``\n            The name of the parent module to re-allow.\n\n        Examples\n        --------\n        >>> logger.disable("__main__")\n        >>> logger.info("Disabled, so nothing is logged.")\n        >>> logger.enable("__main__")\n        >>> logger.info("Re-enabled, messages are logged.")\n        [22:46:12] Re-enabled, messages are logged.\n        '
        self._change_activation(name, True)

    def configure(self, *, handlers=None, levels=None, extra=None, patcher=None, activation=None):
        if False:
            while True:
                i = 10
        'Configure the core logger.\n\n        It should be noted that ``extra`` values set using this function are available across all\n        modules, so this is the best way to set overall default values.\n\n        To load the configuration directly from a file, such as JSON or YAML, it is also possible to\n        use the |loguru-config|_ library developed by `@erezinman`_.\n\n        Parameters\n        ----------\n        handlers : |list| of |dict|, optional\n            A list of each handler to be added. The list should contain dicts of params passed to\n            the |add| function as keyword arguments. If not ``None``, all previously added\n            handlers are first removed.\n        levels : |list| of |dict|, optional\n            A list of each level to be added or updated. The list should contain dicts of params\n            passed to the |level| function as keyword arguments. This will never remove previously\n            created levels.\n        extra : |dict|, optional\n            A dict containing additional parameters bound to the core logger, useful to share\n            common properties if you call |bind| in several of your files modules. If not ``None``,\n            this will remove previously configured ``extra`` dict.\n        patcher : |callable|_, optional\n            A function that will be applied to the record dict of each logged messages across all\n            modules using the logger. It should modify the dict in-place without returning anything.\n            The function is executed prior to the one possibly added by the |patch| method. If not\n            ``None``, this will replace previously configured ``patcher`` function.\n        activation : |list| of |tuple|, optional\n            A list of ``(name, state)`` tuples which denotes which loggers should be enabled (if\n            ``state`` is ``True``) or disabled (if ``state`` is ``False``). The calls to |enable|\n            and |disable| are made accordingly to the list order. This will not modify previously\n            activated loggers, so if you need a fresh start prepend your list with ``("", False)``\n            or ``("", True)``.\n\n        Returns\n        -------\n        :class:`list` of :class:`int`\n            A list containing the identifiers of added sinks (if any).\n\n        Examples\n        --------\n        >>> logger.configure(\n        ...     handlers=[\n        ...         dict(sink=sys.stderr, format="[{time}] {message}"),\n        ...         dict(sink="file.log", enqueue=True, serialize=True),\n        ...     ],\n        ...     levels=[dict(name="NEW", no=13, icon="¤", color="")],\n        ...     extra={"common_to_all": "default"},\n        ...     patcher=lambda record: record["extra"].update(some_value=42),\n        ...     activation=[("my_module.secret", False), ("another_library.module", True)],\n        ... )\n        [1, 2]\n\n        >>> # Set a default "extra" dict to logger across all modules, without "bind()"\n        >>> extra = {"context": "foo"}\n        >>> logger.configure(extra=extra)\n        >>> logger.add(sys.stderr, format="{extra[context]} - {message}")\n        >>> logger.info("Context without bind")\n        >>> # => "foo - Context without bind"\n        >>> logger.bind(context="bar").info("Suppress global context")\n        >>> # => "bar - Suppress global context"\n        '
        if handlers is not None:
            self.remove()
        else:
            handlers = []
        if levels is not None:
            for params in levels:
                self.level(**params)
        if patcher is not None:
            with self._core.lock:
                self._core.patcher = patcher
        if extra is not None:
            with self._core.lock:
                self._core.extra.clear()
                self._core.extra.update(extra)
        if activation is not None:
            for (name, state) in activation:
                if state:
                    self.enable(name)
                else:
                    self.disable(name)
        return [self.add(**params) for params in handlers]

    def _change_activation(self, name, status):
        if False:
            for i in range(10):
                print('nop')
        if not (name is None or isinstance(name, str)):
            raise TypeError("Invalid name, it should be a string (or None), not: '%s'" % type(name).__name__)
        with self._core.lock:
            enabled = self._core.enabled.copy()
            if name is None:
                for n in enabled:
                    if n is None:
                        enabled[n] = status
                self._core.activation_none = status
                self._core.enabled = enabled
                return
            if name != '':
                name += '.'
            activation_list = [(n, s) for (n, s) in self._core.activation_list if n[:len(name)] != name]
            parent_status = next((s for (n, s) in activation_list if name[:len(n)] == n), None)
            if parent_status != status and (not (name == '' and status is True)):
                activation_list.append((name, status))

                def modules_depth(x):
                    if False:
                        print('Hello World!')
                    return x[0].count('.')
                activation_list.sort(key=modules_depth, reverse=True)
            for n in enabled:
                if n is not None and (n + '.')[:len(name)] == name:
                    enabled[n] = status
            self._core.activation_list = activation_list
            self._core.enabled = enabled

    @staticmethod
    def parse(file, pattern, *, cast={}, chunk=2 ** 16):
        if False:
            i = 10
            return i + 15
        'Parse raw logs and extract each entry as a |dict|.\n\n        The logging format has to be specified as the regex ``pattern``, it will then be\n        used to parse the ``file`` and retrieve each entry based on the named groups present\n        in the regex.\n\n        Parameters\n        ----------\n        file : |str|, |Path| or |file-like object|_\n            The path of the log file to be parsed, or an already opened file object.\n        pattern : |str| or |re.Pattern|_\n            The regex to use for logs parsing, it should contain named groups which will be included\n            in the returned dict.\n        cast : |callable|_ or |dict|, optional\n            A function that should convert in-place the regex groups parsed (a dict of string\n            values) to more appropriate types. If a dict is passed, it should be a mapping between\n            keys of parsed log dict and the function that should be used to convert the associated\n            value.\n        chunk : |int|, optional\n            The number of bytes read while iterating through the logs, this avoids having to load\n            the whole file in memory.\n\n        Yields\n        ------\n        :class:`dict`\n            The dict mapping regex named groups to matched values, as returned by |match.groupdict|\n            and optionally converted according to ``cast`` argument.\n\n        Examples\n        --------\n        >>> reg = r"(?P<lvl>[0-9]+): (?P<msg>.*)"    # If log format is "{level.no} - {message}"\n        >>> for e in logger.parse("file.log", reg):  # A file line could be "10 - A debug message"\n        ...     print(e)                             # => {\'lvl\': \'10\', \'msg\': \'A debug message\'}\n\n        >>> caster = dict(lvl=int)                   # Parse \'lvl\' key as an integer\n        >>> for e in logger.parse("file.log", reg, cast=caster):\n        ...     print(e)                             # => {\'lvl\': 10, \'msg\': \'A debug message\'}\n\n        >>> def cast(groups):\n        ...     if "date" in groups:\n        ...         groups["date"] = datetime.strptime(groups["date"], "%Y-%m-%d %H:%M:%S")\n        ...\n        >>> with open("file.log") as file:\n        ...     for log in logger.parse(file, reg, cast=cast):\n        ...         print(log["date"], log["something_else"])\n        '
        if isinstance(file, (str, PathLike)):
            should_close = True
            fileobj = open(str(file))
        elif hasattr(file, 'read') and callable(file.read):
            should_close = False
            fileobj = file
        else:
            raise TypeError("Invalid file, it should be a string path or a file object, not: '%s'" % type(file).__name__)
        if isinstance(cast, dict):

            def cast_function(groups):
                if False:
                    while True:
                        i = 10
                for (key, converter) in cast.items():
                    if key in groups:
                        groups[key] = converter(groups[key])
        elif callable(cast):
            cast_function = cast
        else:
            raise TypeError("Invalid cast, it should be a function or a dict, not: '%s'" % type(cast).__name__)
        try:
            regex = re.compile(pattern)
        except TypeError:
            raise TypeError("Invalid pattern, it should be a string or a compiled regex, not: '%s'" % type(pattern).__name__) from None
        matches = Logger._find_iter(fileobj, regex, chunk)
        for match in matches:
            groups = match.groupdict()
            cast_function(groups)
            yield groups
        if should_close:
            fileobj.close()

    @staticmethod
    def _find_iter(fileobj, regex, chunk):
        if False:
            while True:
                i = 10
        buffer = fileobj.read(0)
        while 1:
            text = fileobj.read(chunk)
            buffer += text
            matches = list(regex.finditer(buffer))
            if not text:
                yield from matches
                break
            if len(matches) > 1:
                end = matches[-2].end()
                buffer = buffer[end:]
                yield from matches[:-1]

    def _log(self, level, from_decorator, options, message, args, kwargs):
        if False:
            i = 10
            return i + 15
        core = self._core
        if not core.handlers:
            return
        try:
            (level_id, level_name, level_no, level_icon) = core.levels_lookup[level]
        except (KeyError, TypeError):
            if isinstance(level, str):
                raise ValueError("Level '%s' does not exist" % level) from None
            if not isinstance(level, int):
                raise TypeError("Invalid level, it should be an integer or a string, not: '%s'" % type(level).__name__) from None
            if level < 0:
                raise ValueError('Invalid level value, it should be a positive integer, not: %d' % level) from None
            cache = (None, 'Level %d' % level, level, ' ')
            (level_id, level_name, level_no, level_icon) = cache
            core.levels_lookup[level] = cache
        if level_no < core.min_level:
            return
        (exception, depth, record, lazy, colors, raw, capture, patchers, extra) = options
        frame = get_frame(depth + 2)
        try:
            name = frame.f_globals['__name__']
        except KeyError:
            name = None
        try:
            if not core.enabled[name]:
                return
        except KeyError:
            enabled = core.enabled
            if name is None:
                status = core.activation_none
                enabled[name] = status
                if not status:
                    return
            else:
                dotted_name = name + '.'
                for (dotted_module_name, status) in core.activation_list:
                    if dotted_name[:len(dotted_module_name)] == dotted_module_name:
                        if status:
                            break
                        enabled[name] = False
                        return
                enabled[name] = True
        current_datetime = aware_now()
        code = frame.f_code
        file_path = code.co_filename
        file_name = basename(file_path)
        thread = current_thread()
        process = current_process()
        elapsed = current_datetime - start_time
        if exception:
            if isinstance(exception, BaseException):
                (type_, value, traceback) = (type(exception), exception, exception.__traceback__)
            elif isinstance(exception, tuple):
                (type_, value, traceback) = exception
            else:
                (type_, value, traceback) = sys.exc_info()
            exception = RecordException(type_, value, traceback)
        else:
            exception = None
        log_record = {'elapsed': elapsed, 'exception': exception, 'extra': {**core.extra, **context.get(), **extra}, 'file': RecordFile(file_name, file_path), 'function': code.co_name, 'level': RecordLevel(level_name, level_no, level_icon), 'line': frame.f_lineno, 'message': str(message), 'module': splitext(file_name)[0], 'name': name, 'process': RecordProcess(process.ident, process.name), 'thread': RecordThread(thread.ident, thread.name), 'time': current_datetime}
        if lazy:
            args = [arg() for arg in args]
            kwargs = {key: value() for (key, value) in kwargs.items()}
        if capture and kwargs:
            log_record['extra'].update(kwargs)
        if record:
            if 'record' in kwargs:
                raise TypeError("The message can't be formatted: 'record' shall not be used as a keyword argument while logger has been configured with '.opt(record=True)'")
            kwargs.update(record=log_record)
        if colors:
            if args or kwargs:
                colored_message = Colorizer.prepare_message(message, args, kwargs)
            else:
                colored_message = Colorizer.prepare_simple_message(str(message))
            log_record['message'] = colored_message.stripped
        elif args or kwargs:
            colored_message = None
            log_record['message'] = message.format(*args, **kwargs)
        else:
            colored_message = None
        if core.patcher:
            core.patcher(log_record)
        for patcher in patchers:
            patcher(log_record)
        for handler in core.handlers.values():
            handler.emit(log_record, level_id, from_decorator, raw, colored_message)

    def trace(__self, __message, *args, **kwargs):
        if False:
            return 10
        "Log ``message.format(*args, **kwargs)`` with severity ``'TRACE'``."
        __self._log('TRACE', False, __self._options, __message, args, kwargs)

    def debug(__self, __message, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Log ``message.format(*args, **kwargs)`` with severity ``'DEBUG'``."
        __self._log('DEBUG', False, __self._options, __message, args, kwargs)

    def info(__self, __message, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Log ``message.format(*args, **kwargs)`` with severity ``'INFO'``."
        __self._log('INFO', False, __self._options, __message, args, kwargs)

    def success(__self, __message, *args, **kwargs):
        if False:
            print('Hello World!')
        "Log ``message.format(*args, **kwargs)`` with severity ``'SUCCESS'``."
        __self._log('SUCCESS', False, __self._options, __message, args, kwargs)

    def warning(__self, __message, *args, **kwargs):
        if False:
            print('Hello World!')
        "Log ``message.format(*args, **kwargs)`` with severity ``'WARNING'``."
        __self._log('WARNING', False, __self._options, __message, args, kwargs)

    def error(__self, __message, *args, **kwargs):
        if False:
            return 10
        "Log ``message.format(*args, **kwargs)`` with severity ``'ERROR'``."
        __self._log('ERROR', False, __self._options, __message, args, kwargs)

    def critical(__self, __message, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Log ``message.format(*args, **kwargs)`` with severity ``'CRITICAL'``."
        __self._log('CRITICAL', False, __self._options, __message, args, kwargs)

    def exception(__self, __message, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Convenience method for logging an ``'ERROR'`` with exception information."
        options = (True,) + __self._options[1:]
        __self._log('ERROR', False, options, __message, args, kwargs)

    def log(__self, __level, __message, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Log ``message.format(*args, **kwargs)`` with severity ``level``.'
        __self._log(__level, False, __self._options, __message, args, kwargs)

    def start(self, *args, **kwargs):
        if False:
            return 10
        'Deprecated function to |add| a new handler.\n\n        Warnings\n        --------\n        .. deprecated:: 0.2.2\n          ``start()`` will be removed in Loguru 1.0.0, it is replaced by ``add()`` which is a less\n          confusing name.\n        '
        warnings.warn("The 'start()' method is deprecated, please use 'add()' instead", DeprecationWarning, stacklevel=2)
        return self.add(*args, **kwargs)

    def stop(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Deprecated function to |remove| an existing handler.\n\n        Warnings\n        --------\n        .. deprecated:: 0.2.2\n          ``stop()`` will be removed in Loguru 1.0.0, it is replaced by ``remove()`` which is a less\n          confusing name.\n        '
        warnings.warn("The 'stop()' method is deprecated, please use 'remove()' instead", DeprecationWarning, stacklevel=2)
        return self.remove(*args, **kwargs)