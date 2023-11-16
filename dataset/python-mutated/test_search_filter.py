import pytest
import json
from rest_framework.test import APIRequestFactory
from awx.api.views import HostList
from awx.main.models import Host, Group, Inventory
from awx.api.versioning import reverse

@pytest.mark.django_db
class TestSearchFilter:

    def test_related_research_filter_relation(self, admin):
        if False:
            return 10
        inv = Inventory.objects.create(name='inv')
        group1 = Group.objects.create(name='g1', inventory=inv)
        group2 = Group.objects.create(name='g2', inventory=inv)
        host1 = Host.objects.create(name='host1', inventory=inv)
        host2 = Host.objects.create(name='host2', inventory=inv)
        host3 = Host.objects.create(name='host3', inventory=inv)
        host1.groups.add(group1)
        host2.groups.add(group1)
        host2.groups.add(group2)
        host3.groups.add(group2)
        host1.save()
        host2.save()
        host3.save()
        factory = APIRequestFactory()
        host_list_url = reverse('api:host_list')
        request = factory.get(host_list_url, data={'groups__search': ['g1', 'g2']})
        request.user = admin
        response = HostList.as_view()(request)
        response.render()
        result = json.loads(response.content)
        assert result['count'] == 3
        expected_hosts = ['host1', 'host2', 'host3']
        for i in result['results']:
            expected_hosts.remove(i['name'])
        assert not expected_hosts
        request = factory.get(host_list_url, data={'groups__search': ['g1,g2']})
        request.user = admin
        response = HostList.as_view()(request)
        response.render()
        result = json.loads(response.content)
        assert result['count'] == 1
        assert result['results'][0]['name'] == 'host2'