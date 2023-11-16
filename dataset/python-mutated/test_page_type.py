import graphene
from ....tests.utils import assert_no_permission, get_graphql_content, get_graphql_content_from_response
PAGE_TYPE_QUERY = '\n    query PageType(\n        $id: ID!, $filters: AttributeFilterInput, $where: AttributeWhereInput\n    ) {\n        pageType(id: $id) {\n            id\n            name\n            slug\n            hasPages\n            attributes {\n                slug\n            }\n            availableAttributes(first: 10, filter: $filters, where: $where) {\n                edges {\n                    node {\n                        slug\n                    }\n                }\n            }\n        }\n    }\n'

def test_page_type_query_by_staff(staff_api_client, page_type, author_page_attribute, permission_manage_pages, color_attribute, page):
    if False:
        print('Hello World!')
    staff_user = staff_api_client.user
    staff_user.user_permissions.add(permission_manage_pages)
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = staff_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    content = get_graphql_content(response)
    page_type_data = content['data']['pageType']
    assert page_type_data['slug'] == page_type.slug
    assert page_type_data['name'] == page_type.name
    assert {attr['slug'] for attr in page_type_data['attributes']} == {attr.slug for attr in page_type.page_attributes.all()}
    assert page_type_data['hasPages'] is True
    available_attributes = page_type_data['availableAttributes']['edges']
    assert len(available_attributes) == 1
    assert available_attributes[0]['node']['slug'] == author_page_attribute.slug

def test_page_type_query_by_staff_with_page_type_permission(staff_api_client, page_type, author_page_attribute, permission_manage_page_types_and_attributes, color_attribute, page):
    if False:
        while True:
            i = 10
    staff_user = staff_api_client.user
    staff_user.user_permissions.add(permission_manage_page_types_and_attributes)
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = staff_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    content = get_graphql_content(response)
    page_type_data = content['data']['pageType']
    assert page_type_data['slug'] == page_type.slug
    assert page_type_data['name'] == page_type.name
    assert {attr['slug'] for attr in page_type_data['attributes']} == {attr.slug for attr in page_type.page_attributes.all()}
    assert page_type_data['hasPages'] is True
    available_attributes = page_type_data['availableAttributes']['edges']
    assert len(available_attributes) == 1
    assert available_attributes[0]['node']['slug'] == author_page_attribute.slug

def test_page_type_query_by_staff_no_perm(staff_api_client, page_type, author_page_attribute):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = staff_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    assert_no_permission(response)

def test_page_type_query_by_app(app_api_client, page_type, author_page_attribute, permission_manage_pages, color_attribute):
    if False:
        return 10
    staff_user = app_api_client.app
    staff_user.permissions.add(permission_manage_pages)
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = app_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    content = get_graphql_content(response)
    page_type_data = content['data']['pageType']
    assert page_type_data['slug'] == page_type.slug
    assert page_type_data['name'] == page_type.name
    assert {attr['slug'] for attr in page_type_data['attributes']} == {attr.slug for attr in page_type.page_attributes.all()}
    available_attributes = page_type_data['availableAttributes']['edges']
    assert len(available_attributes) == 1
    assert available_attributes[0]['node']['slug'] == author_page_attribute.slug

def test_page_type_query_by_app_no_perm(app_api_client, page_type, author_page_attribute, permission_manage_page_types_and_attributes):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = app_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    assert_no_permission(response)

def test_staff_query_page_type_by_invalid_id(staff_api_client, page_type):
    if False:
        while True:
            i = 10
    id = 'bh/'
    variables = {'id': id}
    response = staff_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    content = get_graphql_content_from_response(response)
    assert len(content['errors']) == 1
    assert content['errors'][0]['message'] == f'Invalid ID: {id}. Expected: PageType.'
    assert content['data']['pageType'] is None

def test_staff_query_page_type_with_invalid_object_type(staff_api_client, page_type):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('Order', page_type.pk)}
    response = staff_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    content = get_graphql_content(response)
    assert content['data']['pageType'] is None

def test_page_type_query_filter_unassigned_attributes(staff_api_client, page_type, permission_manage_pages, page_type_attribute_list, color_attribute):
    if False:
        for i in range(10):
            print('nop')
    staff_user = staff_api_client.user
    staff_user.user_permissions.add(permission_manage_pages)
    expected_attribute = page_type_attribute_list[0]
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk), 'filters': {'search': expected_attribute.name}}
    response = staff_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    content = get_graphql_content(response)
    page_type_data = content['data']['pageType']
    assert page_type_data['slug'] == page_type.slug
    assert {attr['slug'] for attr in page_type_data['attributes']} == {attr.slug for attr in page_type.page_attributes.all()}
    available_attributes = page_type_data['availableAttributes']['edges']
    assert len(available_attributes) == 1
    assert available_attributes[0]['node']['slug'] == expected_attribute.slug

def test_page_type_query_where_filter_unassigned_attributes(staff_api_client, page_type, permission_manage_pages, page_type_attribute_list, color_attribute):
    if False:
        for i in range(10):
            print('nop')
    staff_user = staff_api_client.user
    staff_user.user_permissions.add(permission_manage_pages)
    expected_attribute = page_type_attribute_list[0]
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk), 'where': {'name': {'eq': expected_attribute.name}}}
    response = staff_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    content = get_graphql_content(response)
    page_type_data = content['data']['pageType']
    assert page_type_data['slug'] == page_type.slug
    assert {attr['slug'] for attr in page_type_data['attributes']} == {attr.slug for attr in page_type.page_attributes.all()}
    available_attributes = page_type_data['availableAttributes']['edges']
    assert len(available_attributes) == 1
    assert available_attributes[0]['node']['slug'] == expected_attribute.slug

def test_page_type_query_no_pages(staff_api_client, page_type, author_page_attribute, permission_manage_pages, color_attribute):
    if False:
        return 10
    staff_user = staff_api_client.user
    staff_user.user_permissions.add(permission_manage_pages)
    variables = {'id': graphene.Node.to_global_id('PageType', page_type.pk)}
    response = staff_api_client.post_graphql(PAGE_TYPE_QUERY, variables)
    content = get_graphql_content(response)
    page_type_data = content['data']['pageType']
    assert page_type_data['slug'] == page_type.slug
    assert page_type_data['name'] == page_type.name
    assert {attr['slug'] for attr in page_type_data['attributes']} == {attr.slug for attr in page_type.page_attributes.all()}
    assert page_type_data['hasPages'] is False
    available_attributes = page_type_data['availableAttributes']['edges']
    assert len(available_attributes) == 1
    assert available_attributes[0]['node']['slug'] == author_page_attribute.slug

def test_query_page_types_for_federation(api_client, page_type):
    if False:
        while True:
            i = 10
    page_type_id = graphene.Node.to_global_id('PageType', page_type.pk)
    variables = {'representations': [{'__typename': 'PageType', 'id': page_type_id}]}
    query = '\n      query GetPageTypeInFederation($representations: [_Any]) {\n        _entities(representations: $representations) {\n          __typename\n          ... on PageType {\n            id\n            name\n          }\n        }\n      }\n    '
    response = api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    assert content['data']['_entities'] == [{'__typename': 'PageType', 'id': page_type_id, 'name': page_type.name}]