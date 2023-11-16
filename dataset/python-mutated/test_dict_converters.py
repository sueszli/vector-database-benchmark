from __future__ import annotations
from units.mock.procenv import ModuleTestCase
import builtins
realimport = builtins.__import__

class TestTextifyContainers(ModuleTestCase):

    def test_module_utils_basic_json_dict_converters(self):
        if False:
            for i in range(10):
                print('nop')
        from ansible.module_utils.basic import json_dict_unicode_to_bytes, json_dict_bytes_to_unicode
        test_data = dict(item1=u'Fóo', item2=[u'Bár', u'Bam'], item3=dict(sub1=u'Súb'), item4=(u'föo', u'bär', u'©'), item5=42)
        res = json_dict_unicode_to_bytes(test_data)
        res2 = json_dict_bytes_to_unicode(res)
        self.assertEqual(test_data, res2)