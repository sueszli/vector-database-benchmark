import graphene
from .....tax.models import TaxClass
from ....tests.utils import assert_no_permission, get_graphql_content
from ..fragments import TAX_CLASS_FRAGMENT
MUTATION = '\n    mutation TaxClassDelete($id: ID!) {\n        taxClassDelete(id: $id) {\n            errors {\n                field\n                message\n                code\n            }\n            taxClass {\n                ...TaxClass\n            }\n        }\n    }\n' + TAX_CLASS_FRAGMENT

def _test_no_permissions(api_client):
    if False:
        while True:
            i = 10
    tax_class = TaxClass.objects.first()
    variables = {'id': graphene.Node.to_global_id('TaxClass', tax_class.pk)}
    response = api_client.post_graphql(MUTATION, variables, permissions=[])
    assert_no_permission(response)

def test_no_permission_staff(staff_api_client):
    if False:
        while True:
            i = 10
    _test_no_permissions(staff_api_client)

def test_no_permission_app(app_api_client):
    if False:
        i = 10
        return i + 15
    _test_no_permissions(app_api_client)

def _test_tax_class_delete(api_client, permission_manage_taxes):
    if False:
        while True:
            i = 10
    tax_class = TaxClass.objects.create(name='Test')
    id = graphene.Node.to_global_id('TaxClass', tax_class.pk)
    variables = {'id': id}
    response = api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxClassDelete']
    assert not data['errors']
    assert data['taxClass']['id'] == id

def test_delete_as_staff(staff_api_client, permission_manage_taxes):
    if False:
        return 10
    _test_tax_class_delete(staff_api_client, permission_manage_taxes)

def test_delete_as_app(app_api_client, permission_manage_taxes):
    if False:
        while True:
            i = 10
    _test_tax_class_delete(app_api_client, permission_manage_taxes)