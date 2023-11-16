from .....site.models import Site
from ....tests.utils import get_graphql_content

def test_shop_domain_update(staff_api_client, permission_manage_settings):
    if False:
        for i in range(10):
            print('nop')
    query = '\n        mutation updateSettings($input: SiteDomainInput!) {\n            shopDomainUpdate(input: $input) {\n                shop {\n                    name\n                    domain {\n                        host,\n                    }\n                }\n            }\n        }\n    '
    new_name = 'saleor test store'
    variables = {'input': {'domain': 'lorem-ipsum.com', 'name': new_name}}
    site = Site.objects.get_current()
    assert site.domain != 'lorem-ipsum.com'
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_settings])
    content = get_graphql_content(response)
    data = content['data']['shopDomainUpdate']['shop']
    assert data['domain']['host'] == 'lorem-ipsum.com'
    assert data['name'] == new_name
    site.refresh_from_db()
    assert site.domain == 'lorem-ipsum.com'
    assert site.name == new_name