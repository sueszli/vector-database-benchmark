from __future__ import annotations
import os
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector

class LSBFactCollector(BaseFactCollector):
    name = 'lsb'
    _fact_ids = set()
    STRIP_QUOTES = '\\\'\\"\\\\'

    def _lsb_release_bin(self, lsb_path, module):
        if False:
            i = 10
            return i + 15
        lsb_facts = {}
        if not lsb_path:
            return lsb_facts
        (rc, out, err) = module.run_command([lsb_path, '-a'], errors='surrogate_then_replace')
        if rc != 0:
            return lsb_facts
        for line in out.splitlines():
            if len(line) < 1 or ':' not in line:
                continue
            value = line.split(':', 1)[1].strip()
            if 'LSB Version:' in line:
                lsb_facts['release'] = value
            elif 'Distributor ID:' in line:
                lsb_facts['id'] = value
            elif 'Description:' in line:
                lsb_facts['description'] = value
            elif 'Release:' in line:
                lsb_facts['release'] = value
            elif 'Codename:' in line:
                lsb_facts['codename'] = value
        return lsb_facts

    def _lsb_release_file(self, etc_lsb_release_location):
        if False:
            for i in range(10):
                print('nop')
        lsb_facts = {}
        if not os.path.exists(etc_lsb_release_location):
            return lsb_facts
        for line in get_file_lines(etc_lsb_release_location):
            value = line.split('=', 1)[1].strip()
            if 'DISTRIB_ID' in line:
                lsb_facts['id'] = value
            elif 'DISTRIB_RELEASE' in line:
                lsb_facts['release'] = value
            elif 'DISTRIB_DESCRIPTION' in line:
                lsb_facts['description'] = value
            elif 'DISTRIB_CODENAME' in line:
                lsb_facts['codename'] = value
        return lsb_facts

    def collect(self, module=None, collected_facts=None):
        if False:
            return 10
        facts_dict = {}
        lsb_facts = {}
        if not module:
            return facts_dict
        lsb_path = module.get_bin_path('lsb_release')
        if lsb_path:
            lsb_facts = self._lsb_release_bin(lsb_path, module=module)
        if not lsb_facts:
            lsb_facts = self._lsb_release_file('/etc/lsb-release')
        if lsb_facts and 'release' in lsb_facts:
            lsb_facts['major_release'] = lsb_facts['release'].split('.')[0]
        for (k, v) in lsb_facts.items():
            if v:
                lsb_facts[k] = v.strip(LSBFactCollector.STRIP_QUOTES)
        facts_dict['lsb'] = lsb_facts
        return facts_dict