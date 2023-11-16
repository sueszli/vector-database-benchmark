"""
This is the default nodegroup matcher.
"""
import logging
import salt.loader
import salt.utils.minions
log = logging.getLogger(__name__)

def _load_matchers(opts):
    if False:
        return 10
    "\n    Store matchers in __context__ so they're only loaded once\n    "
    __context__['matchers'] = salt.loader.matchers(opts)

def match(tgt, nodegroups=None, opts=None, minion_id=None):
    if False:
        while True:
            i = 10
    '\n    This is a compatibility matcher and is NOT called when using\n    nodegroups for remote execution, but is called when the nodegroups\n    matcher is used in states\n    '
    if not opts:
        opts = __opts__
    if not nodegroups:
        log.debug('Nodegroup matcher called with no nodegroups.')
        return False
    if tgt in nodegroups:
        if 'matchers' not in __context__:
            _load_matchers(opts)
        return __context__['matchers']['compound_match.match'](salt.utils.minions.nodegroup_comp(tgt, nodegroups))
    return False