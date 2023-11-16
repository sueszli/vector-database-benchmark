from __future__ import absolute_import, division, print_function
__metaclass__ = type
import pytest
from awx.main.models import Organization, Inventory, InventorySource, Project

@pytest.fixture
def base_inventory():
    if False:
        i = 10
        return i + 15
    org = Organization.objects.create(name='test-org')
    inv = Inventory.objects.create(name='test-inv', organization=org)
    return inv

@pytest.fixture
def project(base_inventory):
    if False:
        for i in range(10):
            print('nop')
    return Project.objects.create(name='test-proj', organization=base_inventory.organization, scm_type='git', scm_url='https://github.com/ansible/test-playbooks.git')

@pytest.mark.django_db
def test_inventory_source_create(run_module, admin_user, base_inventory, project):
    if False:
        i = 10
        return i + 15
    source_path = '/var/lib/awx/example_source_path/'
    result = run_module('inventory_source', dict(name='foo', inventory=base_inventory.name, state='present', source='scm', source_path=source_path, source_project=project.name), admin_user)
    assert result.pop('changed', None), result
    inv_src = InventorySource.objects.get(name='foo')
    assert inv_src.inventory == base_inventory
    result.pop('invocation')
    assert result == {'id': inv_src.id, 'name': 'foo'}

@pytest.mark.django_db
def test_create_inventory_source_implied_org(run_module, admin_user):
    if False:
        while True:
            i = 10
    org = Organization.objects.create(name='test-org')
    inv = Inventory.objects.create(name='test-inv', organization=org)
    result = run_module('inventory_source', dict(name='Test Inventory Source', inventory='test-inv', source='ec2', state='present'), admin_user)
    assert result.pop('changed', None), result
    inv_src = InventorySource.objects.get(name='Test Inventory Source')
    assert inv_src.inventory == inv
    result.pop('invocation')
    assert result == {'name': 'Test Inventory Source', 'id': inv_src.id}

@pytest.mark.django_db
def test_create_inventory_source_multiple_orgs(run_module, admin_user):
    if False:
        while True:
            i = 10
    org = Organization.objects.create(name='test-org')
    Inventory.objects.create(name='test-inv', organization=org)
    org2 = Organization.objects.create(name='test-org-number-two')
    inv2 = Inventory.objects.create(name='test-inv', organization=org2)
    result = run_module('inventory_source', dict(name='Test Inventory Source', inventory=inv2.name, organization='test-org-number-two', source='ec2', state='present'), admin_user)
    assert result.pop('changed', None), result
    inv_src = InventorySource.objects.get(name='Test Inventory Source')
    assert inv_src.inventory == inv2
    result.pop('invocation')
    assert result == {'name': 'Test Inventory Source', 'id': inv_src.id}

@pytest.mark.django_db
def test_falsy_value(run_module, admin_user, base_inventory):
    if False:
        while True:
            i = 10
    result = run_module('inventory_source', dict(name='falsy-test', inventory=base_inventory.name, source='ec2', update_on_launch=True), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed', None), result
    inv_src = InventorySource.objects.get(name='falsy-test')
    assert inv_src.update_on_launch is True
    result = run_module('inventory_source', dict(name='falsy-test', inventory=base_inventory.name, source='ec2', update_on_launch=False), admin_user)
    inv_src.refresh_from_db()
    assert inv_src.update_on_launch is False

@pytest.mark.django_db
def test_missing_required_credential(run_module, admin_user, base_inventory):
    if False:
        print('Hello World!')
    result = run_module('inventory_source', dict(name='Test Azure Source', inventory=base_inventory.name, source='azure_rm', state='present'), admin_user)
    assert result.pop('failed', None) is True, result
    assert 'Credential is required for a cloud source' in result.get('msg', '')

@pytest.mark.django_db
def test_source_project_not_for_cloud(run_module, admin_user, base_inventory, project):
    if False:
        return 10
    result = run_module('inventory_source', dict(name='Test ec2 Inventory Source', inventory=base_inventory.name, source='ec2', state='present', source_project=project.name), admin_user)
    assert result.pop('failed', None) is True, result
    assert 'Cannot set source_project if not SCM type' in result.get('msg', '')

@pytest.mark.django_db
def test_source_path_not_for_cloud(run_module, admin_user, base_inventory):
    if False:
        print('Hello World!')
    result = run_module('inventory_source', dict(name='Test ec2 Inventory Source', inventory=base_inventory.name, source='ec2', state='present', source_path='where/am/I'), admin_user)
    assert result.pop('failed', None) is True, result
    assert 'Cannot set source_path if not SCM type' in result.get('msg', '')

@pytest.mark.django_db
def test_scm_source_needs_project(run_module, admin_user, base_inventory):
    if False:
        for i in range(10):
            print('nop')
    result = run_module('inventory_source', dict(name='SCM inventory without project', inventory=base_inventory.name, state='present', source='scm', source_path='/var/lib/awx/example_source_path/'), admin_user)
    assert result.pop('failed', None), result
    assert 'Project required for scm type sources' in result.get('msg', '')