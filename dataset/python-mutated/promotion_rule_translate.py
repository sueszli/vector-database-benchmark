from ...utils import get_graphql_content
PROMOTION_RULE_TRANSLATE_MUTATION = '\nmutation promotionRuleTranslate($id: ID!, $input: PromotionRuleTranslationInput!,\n$languageCode: LanguageCodeEnum!) {\n  promotionRuleTranslate(id: $id, input: $input, languageCode: $languageCode) {\n    errors {\n      field\n      code\n      message\n    }\n    promotionRule {\n      id\n      translation(languageCode: $languageCode) {\n        id\n        language {\n          code\n        }\n        name\n        description\n      }\n    }\n  }\n}\n'

def translate_promotion_rule(staff_api_client, promotion_rule_id, language_code='EN', input=None):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': promotion_rule_id, 'languageCode': language_code, 'input': input}
    response = staff_api_client.post_graphql(PROMOTION_RULE_TRANSLATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['promotionRuleTranslate']['errors'] == []
    data = content['data']['promotionRuleTranslate']['promotionRule']['translation']
    assert data['id'] is not None
    return data