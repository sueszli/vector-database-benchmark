import pytest
from ....tests.utils import get_graphql_content
WEBHOOKS_QUERY = '\nquery {\n    apps(first:100) {\n        edges {\n            node {\n                webhooks {\n                    id\n                    name\n                    targetUrl\n                    isActive\n                    asyncEvents {\n                        name\n                    }\n                    syncEvents {\n                        name\n                    }\n                    events {\n                        name\n                    }\n                }\n            }\n        }\n    }\n}\n'

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_webhooks(staff_api_client, webhook_events, permission_manage_apps, count_queries):
    if False:
        for i in range(10):
            print('nop')
    content = get_graphql_content(staff_api_client.post_graphql(WEBHOOKS_QUERY, permissions=[permission_manage_apps], check_no_permissions=False))
    assert len(content['data']['apps']['edges']) == 4