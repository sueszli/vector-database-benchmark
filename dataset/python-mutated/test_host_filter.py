import pytest
import urllib.parse
from awx.api.versioning import reverse
from awx.main.models import Organization, Host, Group, Inventory

@pytest.fixture
def inventory_structure():
    if False:
        i = 10
        return i + 15
    org = Organization.objects.create(name='org')
    inv = Inventory.objects.create(name='inv', organization=org)
    Host.objects.create(name='host1', inventory=inv)
    Host.objects.create(name='host2', inventory=inv)
    Host.objects.create(name='host3', inventory=inv)
    Group.objects.create(name='g1', inventory=inv)
    Group.objects.create(name='g2', inventory=inv)
    Group.objects.create(name='g3', inventory=inv)

@pytest.mark.django_db
def test_q1(inventory_structure, get, user):
    if False:
        for i in range(10):
            print('nop')

    def evaluate_query(query, expected_hosts):
        if False:
            print('Hello World!')
        url = reverse('api:host_list')
        get_params = '?host_filter=%s' % urllib.parse.quote(query, safe='')
        response = get(url + get_params, user('admin', True))
        hosts = response.data['results']
        assert len(expected_hosts) == len(hosts)
        host_ids = [host['id'] for host in hosts]
        for (i, expected_host) in enumerate(expected_hosts):
            assert expected_host.id in host_ids
    hosts = Host.objects.all()
    groups = Group.objects.all()
    groups[0].hosts.add(hosts[0], hosts[1])
    groups[1].hosts.add(hosts[0], hosts[1], hosts[2])
    query = '(name="host1" and groups__name="g1")'
    evaluate_query(query, [hosts[0]])
    query = '(name="host1" and groups__name="g1") or (name="host3" and groups__name="g2")'
    evaluate_query(query, [hosts[0], hosts[2]])
    query = 'search="HOST1"'
    evaluate_query(query, [hosts[0]])