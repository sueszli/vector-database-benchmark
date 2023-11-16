from unittest.mock import patch
import graphene
from ....core.enums import LanguageCodeEnum, TranslationErrorCode
from ....tests.utils import get_graphql_content
PRODUCT_VARIANT_BULK_TRANSLATE_MUTATION = '\n    mutation ProductVariantBulkTranslate(\n        $translations: [ProductVariantBulkTranslateInput!]!\n    ) {\n        productVariantBulkTranslate(translations: $translations) {\n            results {\n                errors {\n                    path\n                    code\n                    message\n                }\n                translation {\n                    id\n                    name\n                    language {\n                        code\n                    }\n                }\n            }\n            count\n        }\n    }\n'

@patch('saleor.plugins.manager.PluginsManager.translation_created')
def test_product_variant_variant_bulk_translate_creates_translations(created_webhook_mock, staff_api_client, variant, permission_manage_translations, settings):
    if False:
        while True:
            i = 10
    assert variant.translations.count() == 0
    variant_global_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    translations = [{'id': variant_global_id, 'languageCode': LanguageCodeEnum.PL.name, 'translationFields': {'name': 'Product PL'}}, {'id': variant_global_id, 'languageCode': LanguageCodeEnum.DE.name, 'translationFields': {'name': 'Product DE'}}]
    staff_api_client.user.user_permissions.add(permission_manage_translations)
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_BULK_TRANSLATE_MUTATION, {'translations': translations})
    content = get_graphql_content(response)
    data = content['data']['productVariantBulkTranslate']
    assert variant.translations.count() == 2
    assert not data['results'][0]['errors']
    assert not data['results'][1]['errors']
    assert data['count'] == 2
    assert data['results'][0]['translation']['name'] == 'Product PL'
    assert data['results'][1]['translation']['name'] == 'Product DE'
    assert created_webhook_mock.call_count == 2

@patch('saleor.plugins.manager.PluginsManager.translation_updated')
def test_product_variant_bulk_translate_updates_translations(updated_webhook_mock, staff_api_client, variant_with_translations, permission_manage_translations, settings):
    if False:
        for i in range(10):
            print('nop')
    assert variant_with_translations.translations.count() == 2
    variant_global_id = graphene.Node.to_global_id('ProductVariant', variant_with_translations.id)
    translations = [{'id': variant_global_id, 'languageCode': LanguageCodeEnum.PL.name, 'translationFields': {'name': 'NewVariant PL'}}, {'id': variant_global_id, 'languageCode': LanguageCodeEnum.DE.name, 'translationFields': {'name': 'NewVariant DE'}}]
    staff_api_client.user.user_permissions.add(permission_manage_translations)
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_BULK_TRANSLATE_MUTATION, {'translations': translations})
    content = get_graphql_content(response)
    data = content['data']['productVariantBulkTranslate']
    assert variant_with_translations.translations.count() == 2
    assert not data['results'][0]['errors']
    assert not data['results'][1]['errors']
    assert data['count'] == 2
    assert data['results'][0]['translation']['name'] == 'NewVariant PL'
    assert data['results'][1]['translation']['name'] == 'NewVariant DE'
    assert updated_webhook_mock.call_count == 2

@patch('saleor.plugins.manager.PluginsManager.translation_created')
def test_product_variant_bulk_translate_creates_translations_using_attr_external_ref(created_webhook_mock, staff_api_client, variant, permission_manage_translations, settings):
    if False:
        i = 10
        return i + 15
    assert variant.translations.count() == 0
    variant.external_reference = 'variant-external-reference'
    variant.save(update_fields=['external_reference'])
    translations = [{'externalReference': variant.external_reference, 'languageCode': LanguageCodeEnum.PL.name, 'translationFields': {'name': 'Variant PL'}}, {'externalReference': variant.external_reference, 'languageCode': LanguageCodeEnum.DE.name, 'translationFields': {'name': 'Variant DE'}}]
    staff_api_client.user.user_permissions.add(permission_manage_translations)
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_BULK_TRANSLATE_MUTATION, {'translations': translations})
    content = get_graphql_content(response)
    data = content['data']['productVariantBulkTranslate']
    assert variant.translations.count() == 2
    assert not data['results'][0]['errors']
    assert not data['results'][1]['errors']
    assert data['count'] == 2
    assert data['results'][0]['translation']['name'] == 'Variant PL'
    assert data['results'][1]['translation']['name'] == 'Variant DE'
    assert created_webhook_mock.call_count == 2

@patch('saleor.plugins.manager.PluginsManager.translation_updated')
def test_product_variant_bulk_translate_updates_translations_using_attr_external_ref(updated_webhook_mock, staff_api_client, variant_with_translations, permission_manage_translations, settings):
    if False:
        for i in range(10):
            print('nop')
    assert variant_with_translations.translations.count() == 2
    variant_with_translations.external_reference = 'variant-external-reference'
    variant_with_translations.save(update_fields=['external_reference'])
    translations = [{'externalReference': variant_with_translations.external_reference, 'languageCode': LanguageCodeEnum.PL.name, 'translationFields': {'name': 'NewVariant PL'}}, {'externalReference': variant_with_translations.external_reference, 'languageCode': LanguageCodeEnum.DE.name, 'translationFields': {'name': 'NewVariant DE'}}]
    staff_api_client.user.user_permissions.add(permission_manage_translations)
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_BULK_TRANSLATE_MUTATION, {'translations': translations})
    content = get_graphql_content(response)
    data = content['data']['productVariantBulkTranslate']
    assert variant_with_translations.translations.count() == 2
    assert not data['results'][0]['errors']
    assert not data['results'][1]['errors']
    assert data['count'] == 2
    assert data['results'][0]['translation']['name'] == 'NewVariant PL'
    assert data['results'][1]['translation']['name'] == 'NewVariant DE'
    assert updated_webhook_mock.call_count == 2

def test_product_variant_bulk_translate_return_error_when_attr_id_and_external_ref(staff_api_client, variant, permission_manage_translations, settings):
    if False:
        for i in range(10):
            print('nop')
    assert variant.translations.count() == 0
    variant.external_reference = 'variant-external-reference'
    variant.save(update_fields=['external_reference'])
    variant_global_id = graphene.Node.to_global_id('ProductVariant', variant.id)
    translations = [{'id': variant_global_id, 'externalReference': variant.external_reference, 'languageCode': LanguageCodeEnum.PL.name, 'translationFields': {'name': 'Variant PL'}}]
    staff_api_client.user.user_permissions.add(permission_manage_translations)
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_BULK_TRANSLATE_MUTATION, {'translations': translations})
    content = get_graphql_content(response)
    data = content['data']['productVariantBulkTranslate']
    assert variant.translations.count() == 0
    assert data['count'] == 0
    message = "Argument 'id' cannot be combined with 'externalReference'"
    error = data['results'][0]['errors'][0]
    assert error['code'] == TranslationErrorCode.INVALID.name
    assert error['message'] == message

def test_product_variant_bulk_translate_return_error_when_invalid_attr_id(staff_api_client, permission_manage_translations, settings):
    if False:
        for i in range(10):
            print('nop')
    translations = [{'id': graphene.Node.to_global_id('ProductVariant', -1), 'languageCode': LanguageCodeEnum.PL.name, 'translationFields': {'name': 'Variant PL'}}]
    staff_api_client.user.user_permissions.add(permission_manage_translations)
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_BULK_TRANSLATE_MUTATION, {'translations': translations})
    content = get_graphql_content(response)
    data = content['data']['productVariantBulkTranslate']
    assert data['count'] == 0
    message = "Couldn't resolve to an object."
    error = data['results'][0]['errors'][0]
    assert error['code'] == TranslationErrorCode.NOT_FOUND.name
    assert error['message'] == message
    assert error['path'] == 'id'

def test_product_variant_bulk_translate_return_error_when_invalid_attr_external_ref(staff_api_client, permission_manage_translations, settings):
    if False:
        print('Hello World!')
    translations = [{'externalReference': 'invalid_reference', 'languageCode': LanguageCodeEnum.PL.name, 'translationFields': {'name': 'Product PL'}}]
    staff_api_client.user.user_permissions.add(permission_manage_translations)
    response = staff_api_client.post_graphql(PRODUCT_VARIANT_BULK_TRANSLATE_MUTATION, {'translations': translations})
    content = get_graphql_content(response)
    data = content['data']['productVariantBulkTranslate']
    assert data['count'] == 0
    message = "Couldn't resolve to an object."
    error = data['results'][0]['errors'][0]
    assert error['code'] == TranslationErrorCode.NOT_FOUND.name
    assert error['message'] == message
    assert error['path'] == 'externalReference'