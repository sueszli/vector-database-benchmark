import pytest
from ....tests.utils import get_graphql_content
MULTIPLE_CHECKOUT_DETAILS_QUERY = '\nquery multipleCheckouts {\n  checkouts(first: 100){\n    edges {\n      node {\n        id\n        channel {\n          id\n          slug\n        }\n      }\n    }\n  }\n}\n'

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_staff_multiple_checkouts(staff_api_client, permission_manage_checkouts, permission_manage_users, checkouts_for_benchmarks, count_queries):
    if False:
        while True:
            i = 10
    staff_api_client.user.user_permissions.set([permission_manage_checkouts, permission_manage_users])
    content = get_graphql_content(staff_api_client.post_graphql(MULTIPLE_CHECKOUT_DETAILS_QUERY))
    assert len(content['data']['checkouts']['edges']) == 10