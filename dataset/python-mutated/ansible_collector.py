from __future__ import annotations
import fnmatch
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
from ansible.module_utils.facts import collector
from ansible.module_utils.common.collections import is_string

class AnsibleFactCollector(collector.BaseFactCollector):
    """A FactCollector that returns results under 'ansible_facts' top level key.

       If a namespace if provided, facts will be collected under that namespace.
       For ex, a ansible.module_utils.facts.namespace.PrefixFactNamespace(prefix='ansible_')

       Has a 'from_gather_subset() constructor that populates collectors based on a
       gather_subset specifier."""

    def __init__(self, collectors=None, namespace=None, filter_spec=None):
        if False:
            for i in range(10):
                print('nop')
        super(AnsibleFactCollector, self).__init__(collectors=collectors, namespace=namespace)
        self.filter_spec = filter_spec

    def _filter(self, facts_dict, filter_spec):
        if False:
            print('Hello World!')
        if not filter_spec or filter_spec == '*':
            return facts_dict
        if is_string(filter_spec):
            filter_spec = [filter_spec]
        found = []
        for f in filter_spec:
            for (x, y) in facts_dict.items():
                if not f or fnmatch.fnmatch(x, f):
                    found.append((x, y))
                elif not f.startswith(('ansible_', 'facter', 'ohai')):
                    g = 'ansible_%s' % f
                    if fnmatch.fnmatch(x, g):
                        found.append((x, y))
        return found

    def collect(self, module=None, collected_facts=None):
        if False:
            while True:
                i = 10
        collected_facts = collected_facts or {}
        facts_dict = {}
        for collector_obj in self.collectors:
            info_dict = {}
            try:
                info_dict = collector_obj.collect_with_namespace(module=module, collected_facts=collected_facts)
            except Exception as e:
                sys.stderr.write(repr(e))
                sys.stderr.write('\n')
            collected_facts.update(info_dict.copy())
            facts_dict.update(self._filter(info_dict, self.filter_spec))
        return facts_dict

class CollectorMetaDataCollector(collector.BaseFactCollector):
    """Collector that provides a facts with the gather_subset metadata."""
    name = 'gather_subset'
    _fact_ids = set()

    def __init__(self, collectors=None, namespace=None, gather_subset=None, module_setup=None):
        if False:
            for i in range(10):
                print('nop')
        super(CollectorMetaDataCollector, self).__init__(collectors, namespace)
        self.gather_subset = gather_subset
        self.module_setup = module_setup

    def collect(self, module=None, collected_facts=None):
        if False:
            while True:
                i = 10
        meta_facts = {'gather_subset': self.gather_subset}
        if self.module_setup:
            meta_facts['module_setup'] = self.module_setup
        return meta_facts

def get_ansible_collector(all_collector_classes, namespace=None, filter_spec=None, gather_subset=None, gather_timeout=None, minimal_gather_subset=None):
    if False:
        while True:
            i = 10
    filter_spec = filter_spec or []
    gather_subset = gather_subset or ['all']
    gather_timeout = gather_timeout or timeout.DEFAULT_GATHER_TIMEOUT
    minimal_gather_subset = minimal_gather_subset or frozenset()
    collector_classes = collector.collector_classes_from_gather_subset(all_collector_classes=all_collector_classes, minimal_gather_subset=minimal_gather_subset, gather_subset=gather_subset, gather_timeout=gather_timeout)
    collectors = []
    for collector_class in collector_classes:
        collector_obj = collector_class(namespace=namespace)
        collectors.append(collector_obj)
    collector_meta_data_collector = CollectorMetaDataCollector(gather_subset=gather_subset, module_setup=True)
    collectors.append(collector_meta_data_collector)
    fact_collector = AnsibleFactCollector(collectors=collectors, filter_spec=filter_spec, namespace=namespace)
    return fact_collector