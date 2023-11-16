import graphene
from django.test import override_settings
from .....graphql.tests.utils import get_graphql_content
from .....tax.models import TaxClass
PRODUCT_TYPE_CREATE_MUTATION_TAX_CODE = '\n  mutation createProductType($name: String, $taxCode: String) {\n    productTypeCreate(input: {name: $name, taxCode: $taxCode}) {\n      productType {\n        name\n        slug\n        taxClass {\n          id\n          name\n          metadata {\n            key\n            value\n          }\n        }\n      }\n      errors {\n        field\n        message\n        code\n      }\n    }\n  }\n'

@override_settings(PLUGINS=['saleor.plugins.avatax.plugin.AvataxPlugin'])
def test_product_type_create_tax_code_creates_new_tax_class(staff_api_client, permission_manage_product_types_and_attributes, plugin_configuration, monkeypatch):
    if False:
        return 10
    plugin_configuration()
    tax_code = 'P0000000'
    monkeypatch.setattr('saleor.plugins.avatax.plugin.get_cached_tax_codes_or_fetch', lambda _: {tax_code: 'desc'})
    variables = {'name': 'New product type', 'taxCode': tax_code}
    TaxClass.objects.all().delete()
    response = staff_api_client.post_graphql(PRODUCT_TYPE_CREATE_MUTATION_TAX_CODE, variables, permissions=[permission_manage_product_types_and_attributes])
    content = get_graphql_content(response)
    tax_class = TaxClass.objects.first()
    assert tax_class
    assert tax_class.name == tax_code
    assert tax_class.metadata
    assert tax_class.metadata['avatax.code'] == tax_code
    assert content['data']['productTypeCreate']['productType']['taxClass']['id'] == graphene.Node.to_global_id('TaxClass', tax_class.pk)
PRODUCT_TYPE_UPDATE_MUTATION_TAX_CODE = '\n  mutation updateProductType($id: ID!, $taxCode: String) {\n    productTypeUpdate(id: $id, input: {taxCode: $taxCode}) {\n      productType {\n        name\n        slug\n        taxClass {\n          id\n          name\n          metadata {\n            key\n            value\n          }\n        }\n      }\n      errors {\n        field\n        message\n        code\n      }\n    }\n  }\n'

@override_settings(PLUGINS=['saleor.plugins.avatax.plugin.AvataxPlugin'])
def test_product_type_update_tax_code_creates_new_tax_class(staff_api_client, product_type, permission_manage_product_types_and_attributes, plugin_configuration, monkeypatch):
    if False:
        while True:
            i = 10
    plugin_configuration()
    tax_code = 'P0000000'
    monkeypatch.setattr('saleor.plugins.avatax.plugin.get_cached_tax_codes_or_fetch', lambda _: {tax_code: 'desc'})
    variables = {'id': graphene.Node.to_global_id('ProductType', product_type.pk), 'taxCode': tax_code}
    response = staff_api_client.post_graphql(PRODUCT_TYPE_UPDATE_MUTATION_TAX_CODE, variables, permissions=[permission_manage_product_types_and_attributes])
    content = get_graphql_content(response)
    product_type.refresh_from_db()
    tax_class = product_type.tax_class
    assert tax_class
    assert tax_class.metadata['avatax.code'] == tax_code
    assert content['data']['productTypeUpdate']['productType']['taxClass']['id'] == graphene.Node.to_global_id('TaxClass', tax_class.pk)
PRODUCT_CREATE_MUTATION_TAX_CODE = '\n  mutation ProductCreate($name: String!, $productTypeId: ID!, $taxCode: String) {\n    productCreate(\n      input: {name: $name, productType: $productTypeId, taxCode: $taxCode}\n    ) {\n      errors {\n        field\n        message\n      }\n      product {\n        id\n        name\n        taxClass {\n          id\n          name\n          metadata {\n            key\n            value\n          }\n        }\n      }\n    }\n  }\n'

@override_settings(PLUGINS=['saleor.plugins.avatax.plugin.AvataxPlugin'])
def test_product_create_tax_code_creates_new_tax_class(staff_api_client, product_type, permission_manage_products, plugin_configuration, monkeypatch):
    if False:
        return 10
    plugin_configuration()
    tax_code = 'P0000000'
    monkeypatch.setattr('saleor.plugins.avatax.plugin.get_cached_tax_codes_or_fetch', lambda _: {tax_code: 'desc'})
    variables = {'name': 'New product', 'productTypeId': graphene.Node.to_global_id('ProductType', product_type.pk), 'taxCode': tax_code}
    TaxClass.objects.all().delete()
    response = staff_api_client.post_graphql(PRODUCT_CREATE_MUTATION_TAX_CODE, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    tax_class = TaxClass.objects.first()
    assert tax_class
    assert tax_class.name == tax_code
    assert tax_class.metadata
    assert tax_class.metadata['avatax.code'] == tax_code
    assert content['data']['productCreate']['product']['taxClass']['id'] == graphene.Node.to_global_id('TaxClass', tax_class.pk)
PRODUCT_UPDATE_MUTATION_TAX_CODE = '\n  mutation ProductUpdate($id: ID!, $taxCode: String) {\n    productUpdate(id: $id, input: {taxCode: $taxCode}) {\n      errors {\n        field\n        message\n      }\n      product {\n        id\n        name\n        taxClass {\n          id\n          name\n          metadata {\n            key\n            value\n          }\n        }\n      }\n    }\n  }\n'

@override_settings(PLUGINS=['saleor.plugins.avatax.plugin.AvataxPlugin'])
def test_product_update_tax_code_creates_new_tax_class(staff_api_client, permission_manage_products, product, plugin_configuration, monkeypatch):
    if False:
        while True:
            i = 10
    plugin_configuration()
    tax_code = 'P0000000'
    monkeypatch.setattr('saleor.plugins.avatax.plugin.get_cached_tax_codes_or_fetch', lambda _: {tax_code: 'desc'})
    variables = {'id': graphene.Node.to_global_id('Product', product.pk), 'taxCode': tax_code}
    response = staff_api_client.post_graphql(PRODUCT_UPDATE_MUTATION_TAX_CODE, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    product.refresh_from_db()
    tax_class = product.tax_class
    assert tax_class
    assert tax_class.name == tax_code
    assert tax_class.metadata
    assert tax_class.metadata['avatax.code'] == tax_code
    assert content['data']['productUpdate']['product']['taxClass']['id'] == graphene.Node.to_global_id('TaxClass', tax_class.pk)