"""
This is the default list matcher.
"""
import logging
log = logging.getLogger(__name__)

def match(tgt, opts=None, minion_id=None):
    if False:
        print('Hello World!')
    '\n    Determines if this host is on the list\n    '
    if not opts:
        opts = __opts__
    if not minion_id:
        minion_id = opts.get('id')
    try:
        if ',{},'.format(minion_id) in tgt or tgt.startswith(minion_id + ',') or tgt.endswith(',' + minion_id):
            return True
        return minion_id == tgt
    except (AttributeError, TypeError):
        try:
            return minion_id in tgt
        except Exception:
            return False
    log.warning('List matcher unexpectedly did not return, for target %s, this is probably a bug.', tgt)
    return False