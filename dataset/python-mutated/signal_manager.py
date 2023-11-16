import logging
from typing import Callable, Optional, List, Tuple, Dict
from feeluown.utils.dispatch import Signal
logger = logging.getLogger(__name__)

class SignalConnector:

    def __init__(self, symbol: str):
        if False:
            print('Hello World!')
        self._signal: Optional[Signal] = None
        self.symbol = symbol
        self._slot_list: List[Tuple[Callable, Dict]] = []
        self._slot_symbol_list: List[Tuple[str, Dict]] = []

    def bind_signal(self, signal: Signal):
        if False:
            for i in range(10):
                print('nop')
        self._signal = signal

    def connect(self):
        if False:
            while True:
                i = 10
        'Connect all slots.\n        '
        if self._signal is None:
            raise RuntimeError('no signal is bound')
        for (slot, kwargs) in self._slot_list:
            self._signal.connect(slot, **kwargs)
        self._slot_list.clear()
        self._signal.connect(self.slot_symbols_delegate, weak=False)

    def connect_slot(self, slot: Callable, **kwargs):
        if False:
            print('Hello World!')
        if self._signal is not None:
            self._signal.connect(slot, **kwargs)
        else:
            self._slot_list.append((slot, kwargs))

    def connect_slot_symbol(self, slot_symbol: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._slot_symbol_list.append((slot_symbol, kwargs))

    def disconnect_slot_symbol(self, slot_symbol):
        if False:
            for i in range(10):
                print('nop')
        for (i, (symbol, _)) in enumerate(self._slot_symbol_list):
            if symbol == slot_symbol:
                self._slot_symbol_list.pop(i)

    def disconnect_slot(self, slot: Callable):
        if False:
            while True:
                i = 10
        if self._signal is not None:
            self._signal.disconnect(slot)
        else:
            for (i, (s, _)) in enumerate(self._slot_list):
                if s == slot:
                    self._slot_list.pop(i)
                    break

    def slot_symbols_delegate(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        A delegate invoke the slots for the signal.\n\n        Signal.emit => self.slot_symbols_delegate => slots\n        '
        for (slot_symbol, kwargs) in self._slot_symbol_list:
            func = fuoexec_F(slot_symbol)
            if kwargs.get('aioqueue'):
                if Signal.has_aio_support:
                    Signal.aioqueue.sync_q.put_nowait((func, args))
                else:
                    logger.warning('No aio support is available, a slot is ignored.')
            else:
                try:
                    func(*args)
                except:
                    logger.exception('error during calling slot:%s')

class SignalManager:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.initialized = False
        self._app = None
        self.signal_connectors: List[SignalConnector] = []

    def initialize(self, app):
        if False:
            print('Hello World!')
        '\n        Find each signal by signal_symbol and connect slots for them.\n        '
        if self.initialized:
            raise RuntimeError('signals slots manager already initialized')
        self._app = app
        for connector in self.signal_connectors:
            self._init_sc(connector)
        self.initialized = True

    def add(self, signal_symbol: str, slot: Callable, use_symbol: bool, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Add one slot for the signal.\n\n        :param slot: The function or it's symbol.\n        "
        sc = self._get_or_create_sc(signal_symbol)
        if use_symbol is True:
            sc.connect_slot_symbol(fuoexec_S(slot), **kwargs)
        else:
            sc.connect_slot(slot, **kwargs)

    def remove(self, signal_symbol: str, slot: Callable, use_symbol: bool):
        if False:
            i = 10
            return i + 15
        'Remove one slot for signal.\n\n        If slot is not connected, this does nothing.\n        '
        signal_connector = self._get_or_create_sc(signal_symbol)
        if use_symbol is True:
            signal_connector.disconnect_slot_symbol(fuoexec_S(slot))
        else:
            signal_connector.disconnect_slot(slot)

    def _get_or_create_sc(self, signal_symbol) -> SignalConnector:
        if False:
            for i in range(10):
                print('nop')
        'Get or create signal connector.'
        for sc in self.signal_connectors:
            if sc.symbol == signal_symbol:
                signal_connector = sc
                break
        else:
            signal_connector = SignalConnector(signal_symbol)
            if self.initialized:
                self._init_sc(signal_connector)
            self.signal_connectors.append(signal_connector)
        return signal_connector

    def _init_sc(self, sc):
        if False:
            i = 10
            return i + 15
        signal = eval(sc.symbol, {'app': self._app})
        sc.bind_signal(signal)
        sc.connect()
signal_mgr: SignalManager = SignalManager()
from .fuoexec import fuoexec_S, fuoexec_F