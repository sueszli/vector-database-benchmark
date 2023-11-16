import logging
import types
from typing import Any, Callable, Dict, Sequence, TypeVar
from .._abc import Instrument
INSTRUMENT_LOGGER = logging.getLogger('trio.abc.Instrument')
F = TypeVar('F', bound=Callable[..., Any])

def _public(fn: F) -> F:
    if False:
        print('Hello World!')
    return fn

class Instruments(Dict[str, Dict[Instrument, None]]):
    """A collection of `trio.abc.Instrument` organized by hook.

    Instrumentation calls are rather expensive, and we don't want a
    rarely-used instrument (like before_run()) to slow down hot
    operations (like before_task_step()). Thus, we cache the set of
    instruments to be called for each hook, and skip the instrumentation
    call if there's nothing currently installed for that hook.
    """
    __slots__ = ()

    def __init__(self, incoming: Sequence[Instrument]):
        if False:
            i = 10
            return i + 15
        self['_all'] = {}
        for instrument in incoming:
            self.add_instrument(instrument)

    @_public
    def add_instrument(self, instrument: Instrument) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Start instrumenting the current run loop with the given instrument.\n\n        Args:\n          instrument (trio.abc.Instrument): The instrument to activate.\n\n        If ``instrument`` is already active, does nothing.\n\n        '
        if instrument in self['_all']:
            return
        self['_all'][instrument] = None
        try:
            for name in dir(instrument):
                if name.startswith('_'):
                    continue
                try:
                    prototype = getattr(Instrument, name)
                except AttributeError:
                    continue
                impl = getattr(instrument, name)
                if isinstance(impl, types.MethodType) and impl.__func__ is prototype:
                    continue
                self.setdefault(name, {})[instrument] = None
        except:
            self.remove_instrument(instrument)
            raise

    @_public
    def remove_instrument(self, instrument: Instrument) -> None:
        if False:
            print('Hello World!')
        'Stop instrumenting the current run loop with the given instrument.\n\n        Args:\n          instrument (trio.abc.Instrument): The instrument to de-activate.\n\n        Raises:\n          KeyError: if the instrument is not currently active. This could\n              occur either because you never added it, or because you added it\n              and then it raised an unhandled exception and was automatically\n              deactivated.\n\n        '
        self['_all'].pop(instrument)
        for (hookname, instruments) in list(self.items()):
            if instrument in instruments:
                del instruments[instrument]
                if not instruments:
                    del self[hookname]

    def call(self, hookname: str, *args: Any) -> None:
        if False:
            print('Hello World!')
        'Call hookname(*args) on each applicable instrument.\n\n        You must first check whether there are any instruments installed for\n        that hook, e.g.::\n\n            if "before_task_step" in instruments:\n                instruments.call("before_task_step", task)\n        '
        for instrument in list(self[hookname]):
            try:
                getattr(instrument, hookname)(*args)
            except BaseException:
                self.remove_instrument(instrument)
                INSTRUMENT_LOGGER.exception('Exception raised when calling %r on instrument %r. Instrument has been disabled.', hookname, instrument)