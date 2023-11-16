from st2common import log as logging
from st2common.metrics.base import BaseMetricsDriver
__all__ = ['EchoDriver']
LOG = logging.getLogger(__name__)

class EchoDriver(BaseMetricsDriver):
    """
    Driver which logs / LOG.debugs out each metrics operation which would have been performed.
    """

    def time(self, key, time):
        if False:
            while True:
                i = 10
        LOG.debug('[metrics] time(key=%s, time=%s)' % (key, time))

    def inc_counter(self, key, amount=1):
        if False:
            i = 10
            return i + 15
        LOG.debug('[metrics] counter.incr(%s, %s)' % (key, amount))

    def dec_counter(self, key, amount=1):
        if False:
            i = 10
            return i + 15
        LOG.debug('[metrics] counter.decr(%s, %s)' % (key, amount))

    def set_gauge(self, key, value):
        if False:
            return 10
        LOG.debug('[metrics] set_gauge(%s, %s)' % (key, value))

    def inc_gauge(self, key, amount=1):
        if False:
            return 10
        LOG.debug('[metrics] gauge.incr(%s, %s)' % (key, amount))

    def dec_gauge(self, key, amount=1):
        if False:
            i = 10
            return i + 15
        LOG.debug('[metrics] gauge.decr(%s, %s)' % (key, amount))