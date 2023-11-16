from __future__ import annotations
import json
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts.collector import BaseFactCollector

class FacterFactCollector(BaseFactCollector):
    name = 'facter'
    _fact_ids = set(['facter'])

    def __init__(self, collectors=None, namespace=None):
        if False:
            return 10
        namespace = PrefixFactNamespace(namespace_name='facter', prefix='facter_')
        super(FacterFactCollector, self).__init__(collectors=collectors, namespace=namespace)

    def find_facter(self, module):
        if False:
            i = 10
            return i + 15
        facter_path = module.get_bin_path('facter', opt_dirs=['/opt/puppetlabs/bin'])
        cfacter_path = module.get_bin_path('cfacter', opt_dirs=['/opt/puppetlabs/bin'])
        if cfacter_path is not None:
            facter_path = cfacter_path
        return facter_path

    def run_facter(self, module, facter_path):
        if False:
            for i in range(10):
                print('nop')
        (rc, out, err) = module.run_command(facter_path + ' --puppet --json')
        if rc != 0:
            (rc, out, err) = module.run_command(facter_path + ' --json')
        return (rc, out, err)

    def get_facter_output(self, module):
        if False:
            for i in range(10):
                print('nop')
        facter_path = self.find_facter(module)
        if not facter_path:
            return None
        (rc, out, err) = self.run_facter(module, facter_path)
        if rc != 0:
            return None
        return out

    def collect(self, module=None, collected_facts=None):
        if False:
            i = 10
            return i + 15
        facter_dict = {}
        if not module:
            return facter_dict
        facter_output = self.get_facter_output(module)
        if facter_output is None:
            return facter_dict
        try:
            facter_dict = json.loads(facter_output)
        except Exception:
            pass
        return facter_dict