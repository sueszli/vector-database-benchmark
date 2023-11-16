import six
import json
import copy
from rqalpha.core.events import EVENT, Event
from rqalpha.environment import Environment
from rqalpha.model.instrument import Instrument

class StrategyUniverse(object):

    def __init__(self):
        if False:
            return 10
        self._set = set()
        Environment.get_instance().event_bus.prepend_listener(EVENT.AFTER_TRADING, self._clear_de_listed)

    def get_state(self):
        if False:
            while True:
                i = 10
        return json.dumps(sorted(self._set)).encode('utf-8')

    def set_state(self, state):
        if False:
            print('Hello World!')
        l = json.loads(state.decode('utf-8'))
        self.update(l)

    def update(self, universe):
        if False:
            i = 10
            return i + 15
        if isinstance(universe, (six.string_types, Instrument)):
            universe = [universe]
        new_set = set(universe)
        if new_set != self._set:
            self._set = new_set
            Environment.get_instance().event_bus.publish_event(Event(EVENT.POST_UNIVERSE_CHANGED, universe=self._set))

    def get(self):
        if False:
            i = 10
            return i + 15
        return copy.copy(self._set)

    def _clear_de_listed(self, event):
        if False:
            print('Hello World!')
        de_listed = set()
        env = Environment.get_instance()
        for o in self._set:
            i = env.data_proxy.instrument(o)
            if i.de_listed_date <= env.trading_dt:
                de_listed.add(o)
        if de_listed:
            self._set -= de_listed
            env.event_bus.publish_event(Event(EVENT.POST_UNIVERSE_CHANGED, universe=self._set))