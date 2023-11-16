from __future__ import annotations
from unittest.mock import Mock, patch
from ..base import BaseFactsTest
from ansible.module_utils.facts.system.lsb import LSBFactCollector
lsb_release_a_fedora_output = '\nLSB Version:\t:core-4.1-amd64:core-4.1-noarch:cxx-4.1-amd64:cxx-4.1-noarch:desktop-4.1-amd64:desktop-4.1-noarch:languages-4.1-amd64:languages-4.1-noarch:printing-4.1-amd64:printing-4.1-noarch\nDistributor ID:\tFedora\nDescription:\tFedora release 25 (Twenty Five)\nRelease:\t25\nCodename:\tTwentyFive\n'
etc_lsb_release_ubuntu14 = 'DISTRIB_ID=Ubuntu\nDISTRIB_RELEASE=14.04\nDISTRIB_CODENAME=trusty\nDISTRIB_DESCRIPTION="Ubuntu 14.04.3 LTS"\n'
etc_lsb_release_no_decimal = 'DISTRIB_ID=AwesomeOS\nDISTRIB_RELEASE=11\nDISTRIB_CODENAME=stonehenge\nDISTRIB_DESCRIPTION="AwesomeÖS 11"\n'

class TestLSBFacts(BaseFactsTest):
    __test__ = True
    gather_subset = ['!all', 'lsb']
    valid_subsets = ['lsb']
    fact_namespace = 'ansible_lsb'
    collector_class = LSBFactCollector

    def _mock_module(self):
        if False:
            for i in range(10):
                print('nop')
        mock_module = Mock()
        mock_module.params = {'gather_subset': self.gather_subset, 'gather_timeout': 10, 'filter': '*'}
        mock_module.get_bin_path = Mock(return_value='/usr/bin/lsb_release')
        mock_module.run_command = Mock(return_value=(0, lsb_release_a_fedora_output, ''))
        return mock_module

    def test_lsb_release_bin(self):
        if False:
            i = 10
            return i + 15
        module = self._mock_module()
        fact_collector = self.collector_class()
        facts_dict = fact_collector.collect(module=module)
        self.assertIsInstance(facts_dict, dict)
        self.assertEqual(facts_dict['lsb']['release'], '25')
        self.assertEqual(facts_dict['lsb']['id'], 'Fedora')
        self.assertEqual(facts_dict['lsb']['description'], 'Fedora release 25 (Twenty Five)')
        self.assertEqual(facts_dict['lsb']['codename'], 'TwentyFive')
        self.assertEqual(facts_dict['lsb']['major_release'], '25')

    def test_etc_lsb_release(self):
        if False:
            i = 10
            return i + 15
        module = self._mock_module()
        module.get_bin_path = Mock(return_value=None)
        with patch('ansible.module_utils.facts.system.lsb.os.path.exists', return_value=True):
            with patch('ansible.module_utils.facts.system.lsb.get_file_lines', return_value=etc_lsb_release_ubuntu14.splitlines()):
                fact_collector = self.collector_class()
                facts_dict = fact_collector.collect(module=module)
        self.assertIsInstance(facts_dict, dict)
        self.assertEqual(facts_dict['lsb']['release'], '14.04')
        self.assertEqual(facts_dict['lsb']['id'], 'Ubuntu')
        self.assertEqual(facts_dict['lsb']['description'], 'Ubuntu 14.04.3 LTS')
        self.assertEqual(facts_dict['lsb']['codename'], 'trusty')

    def test_etc_lsb_release_no_decimal_release(self):
        if False:
            print('Hello World!')
        module = self._mock_module()
        module.get_bin_path = Mock(return_value=None)
        with patch('ansible.module_utils.facts.system.lsb.os.path.exists', return_value=True):
            with patch('ansible.module_utils.facts.system.lsb.get_file_lines', return_value=etc_lsb_release_no_decimal.splitlines()):
                fact_collector = self.collector_class()
                facts_dict = fact_collector.collect(module=module)
        self.assertIsInstance(facts_dict, dict)
        self.assertEqual(facts_dict['lsb']['release'], '11')
        self.assertEqual(facts_dict['lsb']['id'], 'AwesomeOS')
        self.assertEqual(facts_dict['lsb']['description'], 'AwesomeÖS 11')
        self.assertEqual(facts_dict['lsb']['codename'], 'stonehenge')