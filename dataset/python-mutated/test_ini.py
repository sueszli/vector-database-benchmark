from __future__ import annotations
import unittest
from ansible.plugins.lookup.ini import _parse_params

class TestINILookup(unittest.TestCase):
    old_style_params_data = (dict(term=u'keyA section=sectionA file=/path/to/file', expected=[u'file=/path/to/file', u'keyA', u'section=sectionA']), dict(term=u'keyB section=sectionB with space file=/path/with/embedded spaces and/file', expected=[u'file=/path/with/embedded spaces and/file', u'keyB', u'section=sectionB with space']), dict(term=u'keyC section=sectionC file=/path/with/equals/cn=com.ansible', expected=[u'file=/path/with/equals/cn=com.ansible', u'keyC', u'section=sectionC']), dict(term=u'keyD section=sectionD file=/path/with space and/equals/cn=com.ansible', expected=[u'file=/path/with space and/equals/cn=com.ansible', u'keyD', u'section=sectionD']), dict(term=u'keyE section=sectionE file=/path/with/unicode/くらとみ/file', expected=[u'file=/path/with/unicode/くらとみ/file', u'keyE', u'section=sectionE']), dict(term=u'keyF section=sectionF file=/path/with/utf 8 and spaces/くらとみ/file', expected=[u'file=/path/with/utf 8 and spaces/くらとみ/file', u'keyF', u'section=sectionF']))

    def test_parse_parameters(self):
        if False:
            i = 10
            return i + 15
        pvals = {'file': '', 'section': '', 'key': '', 'type': '', 're': '', 'default': '', 'encoding': ''}
        for testcase in self.old_style_params_data:
            params = _parse_params(testcase['term'], pvals)
            params.sort()
            self.assertEqual(params, testcase['expected'])