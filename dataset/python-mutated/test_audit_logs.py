from django.urls import reverse
from rest_framework import status

def test_audit_logs_only_makes_two_queries(admin_client, project, environment, feature, feature_state, django_assert_num_queries):
    if False:
        return 10
    url = reverse('api-v1:audit-list')
    with django_assert_num_queries(2):
        res = admin_client.get(url, {'project': project})
    assert res.status_code == status.HTTP_200_OK
    assert res.json()['count'] == 3