"""
This is the default range matcher.
"""
import logging
HAS_RANGE = False
try:
    import seco.range
    HAS_RANGE = True
except ImportError:
    pass
log = logging.getLogger(__name__)

def match(tgt, opts=None, minion_id=None):
    if False:
        while True:
            i = 10
    '\n    Matches based on range cluster\n    '
    if not opts:
        opts = __opts__
    if HAS_RANGE:
        range_ = seco.range.Range(opts['range_server'])
        try:
            return opts['grains']['fqdn'] in range_.expand(tgt)
        except seco.range.RangeException as exc:
            log.debug('Range exception in compound match: %s', exc)
            return False
    return False