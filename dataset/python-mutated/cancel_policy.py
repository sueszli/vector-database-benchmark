import abc
from abc import abstractmethod
from six import with_metaclass
from zipline.gens.sim_engine import SESSION_END

class CancelPolicy(with_metaclass(abc.ABCMeta)):
    """Abstract cancellation policy interface.
    """

    @abstractmethod
    def should_cancel(self, event):
        if False:
            i = 10
            return i + 15
        'Should all open orders be cancelled?\n\n        Parameters\n        ----------\n        event : enum-value\n            An event type, one of:\n              - :data:`zipline.gens.sim_engine.BAR`\n              - :data:`zipline.gens.sim_engine.DAY_START`\n              - :data:`zipline.gens.sim_engine.DAY_END`\n              - :data:`zipline.gens.sim_engine.MINUTE_END`\n\n        Returns\n        -------\n        should_cancel : bool\n            Should all open orders be cancelled?\n        '
        pass

class EODCancel(CancelPolicy):
    """This policy cancels open orders at the end of the day.  For now,
    Zipline will only apply this policy to minutely simulations.

    Parameters
    ----------
    warn_on_cancel : bool, optional
        Should a warning be raised if this causes an order to be cancelled?
    """

    def __init__(self, warn_on_cancel=True):
        if False:
            print('Hello World!')
        self.warn_on_cancel = warn_on_cancel

    def should_cancel(self, event):
        if False:
            print('Hello World!')
        return event == SESSION_END

class NeverCancel(CancelPolicy):
    """Orders are never automatically canceled.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.warn_on_cancel = False

    def should_cancel(self, event):
        if False:
            for i in range(10):
                print('nop')
        return False