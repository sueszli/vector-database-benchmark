from __future__ import annotations
import os
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector

class LoadAvgFactCollector(BaseFactCollector):
    name = 'loadavg'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        if False:
            i = 10
            return i + 15
        facts = {}
        try:
            loadavg = os.getloadavg()
            facts['loadavg'] = {'1m': loadavg[0], '5m': loadavg[1], '15m': loadavg[2]}
        except OSError:
            pass
        return facts