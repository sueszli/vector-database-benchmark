import xml.etree.ElementTree as ET
import pytest
import salt.utils.xmlutil as xml

@pytest.fixture
def xml_doc():
    if False:
        i = 10
        return i + 15
    return ET.fromstring('\n        <domain>\n            <name>test01</name>\n            <memory unit="MiB">1024</memory>\n            <cpu>\n                <topology sockets="1"/>\n            </cpu>\n            <vcpus>\n              <vcpu enabled="yes" id="1"/>\n            </vcpus>\n            <memtune>\n              <hugepages>\n                <page size="128"/>\n              </hugepages>\n            </memtune>\n        </domain>\n    ')

def test_change_xml_text(xml_doc):
    if False:
        i = 10
        return i + 15
    ret = xml.change_xml(xml_doc, {'name': 'test02'}, [{'path': 'name', 'xpath': 'name'}])
    assert ret
    assert 'test02' == xml_doc.find('name').text

def test_change_xml_text_nochange(xml_doc):
    if False:
        while True:
            i = 10
    ret = xml.change_xml(xml_doc, {'name': 'test01'}, [{'path': 'name', 'xpath': 'name'}])
    assert not ret

def test_change_xml_equals_nochange(xml_doc):
    if False:
        print('Hello World!')
    ret = xml.change_xml(xml_doc, {'mem': 1023}, [{'path': 'mem', 'xpath': 'memory', 'get': lambda n: int(n.text), 'equals': lambda o, n: abs(o - n) <= 1}])
    assert not ret

def test_change_xml_text_notdefined(xml_doc):
    if False:
        return 10
    ret = xml.change_xml(xml_doc, {}, [{'path': 'name', 'xpath': 'name'}])
    assert not ret

def test_change_xml_text_removed(xml_doc):
    if False:
        i = 10
        return i + 15
    ret = xml.change_xml(xml_doc, {'name': None}, [{'path': 'name', 'xpath': 'name'}])
    assert ret
    assert xml_doc.find('name') is None

def test_change_xml_text_add(xml_doc):
    if False:
        print('Hello World!')
    ret = xml.change_xml(xml_doc, {'cpu': {'vendor': 'ACME'}}, [{'path': 'cpu:vendor', 'xpath': 'cpu/vendor'}])
    assert ret
    assert 'ACME' == xml_doc.find('cpu/vendor').text

def test_change_xml_convert(xml_doc):
    if False:
        return 10
    ret = xml.change_xml(xml_doc, {'mem': 2}, [{'path': 'mem', 'xpath': 'memory', 'convert': lambda v: v * 1024}])
    assert ret
    assert '2048' == xml_doc.find('memory').text

def test_change_xml_attr(xml_doc):
    if False:
        i = 10
        return i + 15
    ret = xml.change_xml(xml_doc, {'cpu': {'topology': {'cores': 4}}}, [{'path': 'cpu:topology:cores', 'xpath': 'cpu/topology', 'get': lambda n: int(n.get('cores')) if n.get('cores') else None, 'set': lambda n, v: n.set('cores', str(v)), 'del': xml.del_attribute('cores')}])
    assert ret
    assert '4' == xml_doc.find('cpu/topology').get('cores')

def test_change_xml_attr_unchanged(xml_doc):
    if False:
        for i in range(10):
            print('nop')
    ret = xml.change_xml(xml_doc, {'cpu': {'topology': {'sockets': 1}}}, [{'path': 'cpu:topology:sockets', 'xpath': 'cpu/topology', 'get': lambda n: int(n.get('sockets')) if n.get('sockets') else None, 'set': lambda n, v: n.set('sockets', str(v)), 'del': xml.del_attribute('sockets')}])
    assert not ret

def test_change_xml_attr_remove(xml_doc):
    if False:
        i = 10
        return i + 15
    ret = xml.change_xml(xml_doc, {'cpu': {'topology': {'sockets': None}}}, [{'path': 'cpu:topology:sockets', 'xpath': './cpu/topology', 'get': lambda n: int(n.get('sockets')) if n.get('sockets') else None, 'set': lambda n, v: n.set('sockets', str(v)), 'del': xml.del_attribute('sockets')}])
    assert ret
    assert xml_doc.find('cpu') is None

def test_change_xml_not_simple_value(xml_doc):
    if False:
        return 10
    ret = xml.change_xml(xml_doc, {'cpu': {'topology': {'sockets': None}}}, [{'path': 'cpu', 'xpath': 'vcpu', 'get': lambda n: int(n.text)}])
    assert not ret

def test_change_xml_template(xml_doc):
    if False:
        i = 10
        return i + 15
    ret = xml.change_xml(xml_doc, {'cpu': {'vcpus': {2: {'enabled': True}, 4: {'enabled': False}}}}, [{'path': 'cpu:vcpus:{id}:enabled', 'xpath': "vcpus/vcpu[@id='$id']", 'convert': lambda v: 'yes' if v else 'no', 'get': lambda n: n.get('enabled'), 'set': lambda n, v: n.set('enabled', v), 'del': xml.del_attribute('enabled', ['id'])}])
    assert ret
    assert xml_doc.find("vcpus/vcpu[@id='1']") is None
    assert 'yes' == xml_doc.find("vcpus/vcpu[@id='2']").get('enabled')
    assert 'no' == xml_doc.find("vcpus/vcpu[@id='4']").get('enabled')

def test_change_xml_template_remove(xml_doc):
    if False:
        print('Hello World!')
    ret = xml.change_xml(xml_doc, {'cpu': {'vcpus': None}}, [{'path': 'cpu:vcpus:{id}:enabled', 'xpath': "vcpus/vcpu[@id='$id']", 'convert': lambda v: 'yes' if v else 'no', 'get': lambda n: n.get('enabled'), 'set': lambda n, v: n.set('enabled', v), 'del': xml.del_attribute('enabled', ['id'])}])
    assert ret
    assert xml_doc.find('vcpus') is None

def test_change_xml_template_list(xml_doc):
    if False:
        return 10
    ret = xml.change_xml(xml_doc, {'memtune': {'hugepages': [{'size': '1024'}, {'size': '512'}]}}, [{'path': 'memtune:hugepages:{id}:size', 'xpath': 'memtune/hugepages/page[$id]', 'get': lambda n: n.get('size'), 'set': lambda n, v: n.set('size', v), 'del': xml.del_attribute('size')}])
    assert ret
    assert ['1024', '512'] == [n.get('size') for n in xml_doc.findall('memtune/hugepages/page')]

def test_strip_spaces():
    if False:
        return 10
    xml_str = '<domain>\n            <name>test01</name>\n            <memory unit="MiB" >1024</memory>\n        </domain>\n    '
    expected_str = b'<domain><name>test01</name><memory unit="MiB">1024</memory></domain>'
    node = ET.fromstring(xml_str)
    assert expected_str == ET.tostring(xml.strip_spaces(node))