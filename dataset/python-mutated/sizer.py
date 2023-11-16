from __future__ import absolute_import, division, print_function, unicode_literals
from .utils.py3 import with_metaclass
from .metabase import MetaParams

class Sizer(with_metaclass(MetaParams, object)):
    """This is the base class for *Sizers*. Any *sizer* should subclass this
    and override the ``_getsizing`` method

    Member Attribs:

      - ``strategy``: will be set by the strategy in which the sizer is working

        Gives access to the entire api of the strategy, for example if the
        actual data position would be needed in ``_getsizing``::

           position = self.strategy.getposition(data)

      - ``broker``: will be set by the strategy in which the sizer is working

        Gives access to information some complex sizers may need like portfolio
        value, ..
    """
    strategy = None
    broker = None

    def getsizing(self, data, isbuy):
        if False:
            while True:
                i = 10
        comminfo = self.broker.getcommissioninfo(data)
        return self._getsizing(comminfo, self.broker.getcash(), data, isbuy)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if False:
            i = 10
            return i + 15
        'This method has to be overriden by subclasses of Sizer to provide\n        the sizing functionality\n\n        Params:\n          - ``comminfo``: The CommissionInfo instance that contains\n            information about the commission for the data and allows\n            calculation of position value, operation cost, commision for the\n            operation\n\n          - ``cash``: current available cash in the *broker*\n\n          - ``data``: target of the operation\n\n          - ``isbuy``: will be ``True`` for *buy* operations and ``False``\n            for *sell* operations\n\n        The method has to return the actual size (an int) to be executed. If\n        ``0`` is returned nothing will be executed.\n\n        The absolute value of the returned value will be used\n\n        '
        raise NotImplementedError

    def set(self, strategy, broker):
        if False:
            print('Hello World!')
        self.strategy = strategy
        self.broker = broker
SizerBase = Sizer