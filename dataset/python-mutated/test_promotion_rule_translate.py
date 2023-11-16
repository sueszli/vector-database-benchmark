import json
from unittest.mock import ANY, patch
import graphene
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....webhook.event_types import WebhookEventAsyncType
from ....tests.utils import assert_no_permission, get_graphql_content
PROMOTION_RULE_TRANSLATE_MUTATION = '\n    mutation (\n        $id: ID!,\n        $languageCode: LanguageCodeEnum!,\n        $input: PromotionRuleTranslationInput!\n    ) {\n        promotionRuleTranslate(\n            id: $id,\n            languageCode: $languageCode,\n            input: $input\n        ) {\n            promotionRule {\n                translation(languageCode: $languageCode) {\n                    name\n                    description\n                    language {\n                        code\n                    }\n                }\n            }\n            errors {\n                message\n                code\n                field\n            }\n        }\n    }\n'

@freeze_time('2023-06-01 10:00')
@patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_promotion_rule_create_translation(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, promotion_rule, permission_manage_translations, settings, description_json):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    rule_id = graphene.Node.to_global_id('PromotionRule', promotion_rule.id)
    variables = {'id': rule_id, 'languageCode': 'PL', 'input': {'name': 'Polish rule name', 'description': description_json}}
    response = staff_api_client.post_graphql(PROMOTION_RULE_TRANSLATE_MUTATION, variables, permissions=[permission_manage_translations])
    content = get_graphql_content(response)
    data = content['data']['promotionRuleTranslate']
    assert not data['errors']
    translation_data = data['promotionRule']['translation']
    assert translation_data['name'] == 'Polish rule name'
    assert translation_data['description'] == json.dumps(description_json)
    assert translation_data['language']['code'] == 'PL'
    translation = promotion_rule.translations.first()
    mocked_webhook_trigger.assert_called_once_with(None, WebhookEventAsyncType.TRANSLATION_CREATED, [any_webhook], translation, SimpleLazyObject(lambda : staff_api_client.user), legacy_data_generator=ANY)

def test_promotion_rule_update_translation(staff_api_client, promotion_rule, promotion_rule_translation_fr, permission_manage_translations):
    if False:
        while True:
            i = 10
    assert promotion_rule.translations.first().name == 'French promotion rule name'
    rule_id = graphene.Node.to_global_id('PromotionRule', promotion_rule.id)
    updated_name = 'Updated French rule name'
    variables = {'id': rule_id, 'languageCode': 'FR', 'input': {'name': updated_name}}
    response = staff_api_client.post_graphql(PROMOTION_RULE_TRANSLATE_MUTATION, variables, permissions=[permission_manage_translations])
    content = get_graphql_content(response)
    data = content['data']['promotionRuleTranslate']
    assert not data['errors']
    translation_data = data['promotionRule']['translation']
    assert translation_data['name'] == updated_name
    assert translation_data['language']['code'] == 'FR'
    assert promotion_rule.translations.first().name == updated_name

def test_promotion_rule_create_translation_no_permission(staff_api_client, promotion_rule):
    if False:
        return 10
    rule_id = graphene.Node.to_global_id('PromotionRule', promotion_rule.id)
    variables = {'id': rule_id, 'languageCode': 'PL', 'input': {'name': 'Polish rule name'}}
    response = staff_api_client.post_graphql(PROMOTION_RULE_TRANSLATE_MUTATION, variables)
    assert_no_permission(response)

def test_promotion_rule_create_translation_by_translatable_content_id(staff_api_client, promotion_rule, permission_manage_translations):
    if False:
        return 10
    translatable_content_id = graphene.Node.to_global_id('PromotionRuleTranslatableContent', promotion_rule.id)
    variables = {'id': translatable_content_id, 'languageCode': 'PL', 'input': {'name': 'Polish rule name'}}
    response = staff_api_client.post_graphql(PROMOTION_RULE_TRANSLATE_MUTATION, variables, permissions=[permission_manage_translations])
    content = get_graphql_content(response)
    data = content['data']['promotionRuleTranslate']
    assert not data['errors']
    translation_data = data['promotionRule']['translation']
    assert translation_data['name'] == 'Polish rule name'
    assert translation_data['language']['code'] == 'PL'

def test_promotion_rule_create_translation_clear_old_sale_id(staff_api_client, promotion_converted_from_sale, permission_manage_translations):
    if False:
        while True:
            i = 10
    promotion = promotion_converted_from_sale
    assert promotion.old_sale_id
    rule = promotion.rules.first()
    rule_id = graphene.Node.to_global_id('PromotionRule', rule.id)
    variables = {'id': rule_id, 'languageCode': 'PL', 'input': {'name': 'Polish rule name'}}
    response = staff_api_client.post_graphql(PROMOTION_RULE_TRANSLATE_MUTATION, variables, permissions=[permission_manage_translations])
    content = get_graphql_content(response)
    data = content['data']['promotionRuleTranslate']
    assert not data['errors']
    translation_data = data['promotionRule']['translation']
    assert translation_data['name'] == 'Polish rule name'
    assert translation_data['language']['code'] == 'PL'
    promotion.refresh_from_db()
    assert not promotion.old_sale_id