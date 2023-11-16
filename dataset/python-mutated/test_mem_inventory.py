from awx.main.utils.mem_inventory import MemInventory, mem_data_to_dict, dict_to_mem_data
import pytest
import json

@pytest.fixture
def memory_inventory():
    if False:
        return 10
    inventory = MemInventory()
    h = inventory.get_host('my_host')
    h.variables = {'foo': 'bar'}
    g = inventory.get_group('my_group')
    g.variables = {'foobar': 'barfoo'}
    h2 = inventory.get_host('group_host')
    g.add_host(h2)
    return inventory

@pytest.fixture
def JSON_of_inv():
    if False:
        while True:
            i = 10
    return {'_meta': {'hostvars': {'group_host': {}, 'my_host': {'foo': 'bar'}}}, 'all': {'children': ['my_group', 'ungrouped']}, 'my_group': {'hosts': ['group_host'], 'vars': {'foobar': 'barfoo'}}, 'ungrouped': {'hosts': ['my_host']}}

@pytest.fixture
def JSON_with_lists():
    if False:
        return 10
    docs_example = '{\n       "databases"   : {\n           "hosts"   : [ "host1.example.com", "host2.example.com" ],\n           "vars"    : {\n               "a"   : true\n           }\n       },\n       "webservers"  : [ "host2.example.com", "host3.example.com" ],\n       "atlanta"     : {\n           "hosts"   : [ "host1.example.com", "host4.example.com", "host5.example.com" ],\n           "vars"    : {\n               "b"   : false\n           },\n           "children": [ "marietta", "5points" ]\n       },\n       "marietta"    : [ "host6.example.com" ],\n       "5points"     : [ "host7.example.com" ]\n    }'
    return json.loads(docs_example)

@pytest.mark.inventory_import
def test_inventory_create_all_group():
    if False:
        while True:
            i = 10
    inventory = MemInventory()
    assert inventory.all_group.name == 'all'

@pytest.mark.inventory_import
def test_create_child_group():
    if False:
        for i in range(10):
            print('nop')
    inventory = MemInventory()
    g1 = inventory.get_group('g1')
    g2 = inventory.get_group('g2', g1)
    assert g1.children == [g2]
    assert inventory.all_group.children == [g1]
    assert set(inventory.all_group.all_groups.values()) == set([g1, g2])

@pytest.mark.inventory_import
def test_ungrouped_mechanics():
    if False:
        print('Hello World!')
    inventory = MemInventory()
    ug = inventory.get_group('ungrouped')
    assert ug is inventory.all_group

@pytest.mark.inventory_import
def test_convert_memory_to_JSON_with_vars(memory_inventory):
    if False:
        i = 10
        return i + 15
    data = mem_data_to_dict(memory_inventory)
    assert data['_meta']['hostvars']['my_host'] == {'foo': 'bar'}
    assert data['my_group']['vars'] == {'foobar': 'barfoo'}
    assert data['ungrouped']['hosts'] == ['my_host']

@pytest.mark.inventory_import
def test_convert_JSON_to_memory_with_vars(JSON_of_inv):
    if False:
        while True:
            i = 10
    inventory = dict_to_mem_data(JSON_of_inv)
    assert inventory.get_host('my_host').variables == {'foo': 'bar'}
    assert inventory.get_group('my_group').variables == {'foobar': 'barfoo'}
    assert inventory.get_host('group_host') in inventory.get_group('my_group').hosts

@pytest.mark.inventory_import
def test_host_lists_accepted(JSON_with_lists):
    if False:
        while True:
            i = 10
    inventory = dict_to_mem_data(JSON_with_lists)
    assert inventory.get_group('marietta').name == 'marietta'
    h = inventory.get_host('host6.example.com')
    assert h.name == 'host6.example.com'