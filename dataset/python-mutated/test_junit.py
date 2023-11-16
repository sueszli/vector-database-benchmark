import xml.etree.ElementTree as ET
from pathlib import Path
pytest_plugins = 'pytester'
TESTSUITE = '\nfrom hypothesis import given\nfrom hypothesis.strategies import integers\n\n@given(integers())\ndef test_valid(x):\n    assert x == x\n\n@given(integers())\ndef test_invalid(x):\n    assert x != x\n'

def _run_and_get_junit(testdir, *args):
    if False:
        print('Hello World!')
    script = testdir.makepyfile(TESTSUITE)
    testdir.runpytest(script, '--junit-xml=out.xml', *args)
    return ET.parse(Path(testdir.tmpdir) / 'out.xml').getroot()

def _findall_from_root(junit_xml, path):
    if False:
        for i in range(10):
            print('nop')
    if junit_xml.tag == 'testsuites':
        return junit_xml.findall(f'./testsuite/{path}')
    return junit_xml.findall(f'./{path}')

def suite_properties_ok(junit_xml):
    if False:
        for i in range(10):
            print('nop')
    testsuite_props = _findall_from_root(junit_xml, 'properties')
    return len(testsuite_props) == 1 and {prop.get('name') for prop in testsuite_props[0].findall('property')} == {'hypothesis-statistics-test_outputs_valid_xunit2.py::test_valid', 'hypothesis-statistics-test_outputs_valid_xunit2.py::test_invalid'}

def test_outputs_valid_xunit2(testdir):
    if False:
        i = 10
        return i + 15
    junit_xml = _run_and_get_junit(testdir)
    testcase_props = _findall_from_root(junit_xml, 'testcase/properties')
    assert len(testcase_props) == 0
    assert suite_properties_ok(junit_xml)

def test_outputs_valid_xunit2_with_xdist(testdir):
    if False:
        return 10
    junit_xml = _run_and_get_junit(testdir, '-n2')
    testcase_props = _findall_from_root(junit_xml, 'testcase/properties')
    assert len(testcase_props) == 0
    assert not suite_properties_ok(junit_xml)