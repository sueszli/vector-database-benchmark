"""
This is the default cache matcher function.  It only exists for the master,
this is why there is only a ``mmatch()`` but not ``match()``.
"""
import logging
import salt.utils.data
import salt.utils.minions
log = logging.getLogger(__name__)

def mmatch(expr, delimiter, greedy, search_type, regex_match=False, exact_match=False, opts=None):
    if False:
        while True:
            i = 10
    "\n    Helper function to search for minions in master caches\n    If 'greedy' return accepted minions that matched by the condition or absent in the cache.\n    If not 'greedy' return the only minions have cache data and matched by the condition.\n    "
    if not opts:
        opts = __opts__
    ckminions = salt.utils.minions.CkMinions(opts)
    return ckminions._check_cache_minions(expr, delimiter, greedy, search_type, regex_match=regex_match, exact_match=exact_match)