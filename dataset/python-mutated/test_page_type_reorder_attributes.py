import graphene
from .....page.error_codes import PageErrorCode
from .....page.models import PageType
from ....tests.utils import assert_no_permission, get_graphql_content
PAGE_TYPE_REORDER_ATTRIBUTES_MUTATION = '\n    mutation PageTypeReorderAttributes(\n        $pageTypeId: ID!\n        $moves: [ReorderInput!]!\n    ) {\n        pageTypeReorderAttributes(\n            pageTypeId: $pageTypeId\n            moves: $moves\n        ) {\n            pageType {\n                id\n                attributes {\n                    id\n                    slug\n                }\n            }\n            errors {\n                code\n                field\n                message\n                attributes\n            }\n        }\n    }\n'

def test_reorder_page_type_attributes_by_staff(staff_api_client, permission_manage_page_types_and_attributes, page_type_attribute_list):
    if False:
        print('Hello World!')
    staff_api_client.user.user_permissions.add(permission_manage_page_types_and_attributes)
    attributes = page_type_attribute_list
    assert len(attributes) == 3
    page_type = PageType.objects.create(name='Test page type', slug='test-page-type')
    page_type.page_attributes.set(attributes)
    sorted_attributes = list(page_type.page_attributes.order_by())
    assert len(sorted_attributes) == 3
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'moves': [{'id': graphene.Node.to_global_id('Attribute', sorted_attributes[2].pk), 'sortOrder': -2}, {'id': graphene.Node.to_global_id('Attribute', sorted_attributes[0].pk), 'sortOrder': 1}]}
    response = staff_api_client.post_graphql(PAGE_TYPE_REORDER_ATTRIBUTES_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeReorderAttributes']
    errors = data['errors']
    page_type_data = data['pageType']
    assert not errors
    assert len(page_type_data['attributes']) == len(sorted_attributes)
    expected_order = [sorted_attributes[2].pk, sorted_attributes[1].pk, sorted_attributes[0].pk]
    for (attr, expected_pk) in zip(page_type_data['attributes'], expected_order):
        (gql_type, gql_attr_id) = graphene.Node.from_global_id(attr['id'])
        assert gql_type == 'Attribute'
        assert int(gql_attr_id) == expected_pk

def test_reorder_page_type_attributes_by_staff_no_perm(staff_api_client, page_type_attribute_list):
    if False:
        i = 10
        return i + 15
    attributes = page_type_attribute_list
    assert len(attributes) == 3
    page_type = PageType.objects.create(name='Test page type', slug='test-page-type')
    page_type.page_attributes.set(attributes)
    sorted_attributes = list(page_type.page_attributes.order_by())
    assert len(sorted_attributes) == 3
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'moves': [{'id': graphene.Node.to_global_id('Attribute', attributes[0].pk), 'sortOrder': 1}, {'id': graphene.Node.to_global_id('Attribute', attributes[2].pk), 'sortOrder': -1}]}
    response = staff_api_client.post_graphql(PAGE_TYPE_REORDER_ATTRIBUTES_MUTATION, variables)
    assert_no_permission(response)

def test_reorder_page_type_attributes_by_app(app_api_client, permission_manage_page_types_and_attributes, page_type_attribute_list):
    if False:
        while True:
            i = 10
    app_api_client.app.permissions.add(permission_manage_page_types_and_attributes)
    attributes = page_type_attribute_list
    assert len(attributes) == 3
    page_type = PageType.objects.create(name='Test page type', slug='test-page-type')
    page_type.page_attributes.set(attributes)
    sorted_attributes = list(page_type.page_attributes.order_by())
    assert len(sorted_attributes) == 3
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'moves': [{'id': graphene.Node.to_global_id('Attribute', sorted_attributes[2].pk), 'sortOrder': -2}, {'id': graphene.Node.to_global_id('Attribute', sorted_attributes[0].pk), 'sortOrder': 1}]}
    response = app_api_client.post_graphql(PAGE_TYPE_REORDER_ATTRIBUTES_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeReorderAttributes']
    errors = data['errors']
    page_type_data = data['pageType']
    assert not errors
    assert len(page_type_data['attributes']) == len(sorted_attributes)
    expected_order = [sorted_attributes[2].pk, sorted_attributes[1].pk, sorted_attributes[0].pk]
    for (attr, expected_pk) in zip(page_type_data['attributes'], expected_order):
        (gql_type, gql_attr_id) = graphene.Node.from_global_id(attr['id'])
        assert gql_type == 'Attribute'
        assert int(gql_attr_id) == expected_pk

def test_reorder_page_type_attributes_by_app_no_perm(app_api_client, page_type_attribute_list):
    if False:
        print('Hello World!')
    attributes = page_type_attribute_list
    assert len(attributes) == 3
    page_type = PageType.objects.create(name='Test page type', slug='test-page-type')
    page_type.page_attributes.set(attributes)
    sorted_attributes = list(page_type.page_attributes.order_by())
    assert len(sorted_attributes) == 3
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'moves': [{'id': graphene.Node.to_global_id('Attribute', attributes[2].pk), 'sortOrder': -2}, {'id': graphene.Node.to_global_id('Attribute', attributes[0].pk), 'sortOrder': 1}]}
    response = app_api_client.post_graphql(PAGE_TYPE_REORDER_ATTRIBUTES_MUTATION, variables)
    assert_no_permission(response)

def test_reorder_page_type_attributes_invalid_page_type(staff_api_client, permission_manage_page_types_and_attributes, page_type):
    if False:
        print('Hello World!')
    staff_api_client.user.user_permissions.add(permission_manage_page_types_and_attributes)
    attribute = page_type.page_attributes.first()
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', -1), 'moves': [{'id': graphene.Node.to_global_id('Attribute', attribute.pk), 'sortOrder': 1}]}
    response = staff_api_client.post_graphql(PAGE_TYPE_REORDER_ATTRIBUTES_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeReorderAttributes']
    errors = data['errors']
    page_type_data = data['pageType']
    assert not page_type_data
    assert len(errors) == 1
    assert errors[0]['field'] == 'pageTypeId'
    assert errors[0]['code'] == PageErrorCode.NOT_FOUND.name

def test_reorder_page_type_attributes_invalid_attribute_id(staff_api_client, permission_manage_page_types_and_attributes, page_type, color_attribute, size_attribute):
    if False:
        while True:
            i = 10
    staff_api_client.user.user_permissions.add(permission_manage_page_types_and_attributes)
    page_type_attribute = page_type.page_attributes.first()
    color_attribute_id = graphene.Node.to_global_id('Attribute', color_attribute.pk)
    size_attribute_id = graphene.Node.to_global_id('Attribute', size_attribute.pk)
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'moves': [{'id': color_attribute_id, 'sortOrder': 1}, {'id': graphene.Node.to_global_id('Attribute', page_type_attribute.pk), 'sortOrder': 1}, {'id': size_attribute_id, 'sortOrder': 1}]}
    response = staff_api_client.post_graphql(PAGE_TYPE_REORDER_ATTRIBUTES_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['pageTypeReorderAttributes']
    errors = data['errors']
    page_type_data = data['pageType']
    assert not page_type_data
    assert len(errors) == 1
    assert errors[0]['field'] == 'moves'
    assert errors[0]['code'] == PageErrorCode.NOT_FOUND.name
    assert set(errors[0]['attributes']) == {color_attribute_id, size_attribute_id}