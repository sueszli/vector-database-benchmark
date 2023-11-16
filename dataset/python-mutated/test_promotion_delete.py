import graphene
import pytest
from ....tests.utils import get_graphql_content
from ..mutations.test_promotion_delete import PROMOTION_DELETE_MUTATION

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_promotion_delete(staff_api_client, permission_group_manage_discounts, promotion, count_queries):
    if False:
        i = 10
        return i + 15
    staff_api_client.user.groups.add(permission_group_manage_discounts)
    variables = {'id': graphene.Node.to_global_id('Promotion', promotion.id)}
    content = get_graphql_content(staff_api_client.post_graphql(PROMOTION_DELETE_MUTATION, variables))
    data = content['data']['promotionDelete']
    assert data['promotion']