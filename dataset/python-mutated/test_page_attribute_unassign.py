import graphene
from ....tests.utils import assert_no_permission, get_graphql_content
PAGE_UNASSIGN_ATTR_QUERY = '\n    mutation PageAttributeUnassign(\n        $pageTypeId: ID!, $attributeIds: [ID!]!\n    ) {\n        pageAttributeUnassign(\n            pageTypeId: $pageTypeId, attributeIds: $attributeIds\n        ) {\n            pageType {\n                id\n                attributes {\n                    id\n                }\n            }\n            errors {\n                field\n                code\n                message\n                attributes\n            }\n        }\n    }\n'

def test_unassign_attributes_from_page_type_by_staff(staff_api_client, page_type, permission_manage_page_types_and_attributes):
    if False:
        i = 10
        return i + 15
    staff_user = staff_api_client.user
    staff_user.user_permissions.add(permission_manage_page_types_and_attributes)
    attr_count = page_type.page_attributes.count()
    attr_to_unassign = page_type.page_attributes.first()
    attr_to_unassign_id = graphene.Node.to_global_id('Attribute', attr_to_unassign.pk)
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'attributeIds': [attr_to_unassign_id]}
    response = staff_api_client.post_graphql(PAGE_UNASSIGN_ATTR_QUERY, variables)
    content = get_graphql_content(response)
    data = content['data']['pageAttributeUnassign']
    errors = data['errors']
    assert not errors
    assert len(data['pageType']['attributes']) == attr_count - 1
    assert attr_to_unassign_id not in {attr['id'] for attr in data['pageType']['attributes']}

def test_unassign_attributes_from_page_type_by_staff_no_perm(staff_api_client, page_type):
    if False:
        while True:
            i = 10
    attr_to_unassign = page_type.page_attributes.first()
    attr_to_unassign_id = graphene.Node.to_global_id('Attribute', attr_to_unassign.pk)
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'attributeIds': [attr_to_unassign_id]}
    response = staff_api_client.post_graphql(PAGE_UNASSIGN_ATTR_QUERY, variables)
    assert_no_permission(response)

def test_unassign_attributes_from_page_type_by_app(app_api_client, page_type, permission_manage_page_types_and_attributes):
    if False:
        i = 10
        return i + 15
    app = app_api_client.app
    app.permissions.add(permission_manage_page_types_and_attributes)
    attr_count = page_type.page_attributes.count()
    attr_to_unassign = page_type.page_attributes.first()
    attr_to_unassign_id = graphene.Node.to_global_id('Attribute', attr_to_unassign.pk)
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'attributeIds': [attr_to_unassign_id]}
    response = app_api_client.post_graphql(PAGE_UNASSIGN_ATTR_QUERY, variables)
    content = get_graphql_content(response)
    data = content['data']['pageAttributeUnassign']
    errors = data['errors']
    assert not errors
    assert len(data['pageType']['attributes']) == attr_count - 1
    assert attr_to_unassign_id not in {attr['id'] for attr in data['pageType']['attributes']}

def test_unassign_attributes_from_page_type_by_app_no_perm(app_api_client, page_type):
    if False:
        return 10
    attr_to_unassign = page_type.page_attributes.first()
    attr_to_unassign_id = graphene.Node.to_global_id('Attribute', attr_to_unassign.pk)
    variables = {'pageTypeId': graphene.Node.to_global_id('PageType', page_type.pk), 'attributeIds': [attr_to_unassign_id]}
    response = app_api_client.post_graphql(PAGE_UNASSIGN_ATTR_QUERY, variables)
    assert_no_permission(response)