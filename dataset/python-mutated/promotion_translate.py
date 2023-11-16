from ...utils import get_graphql_content
PROMOTION_TRANSLATE_MUTATION = '\nmutation PromotionTranslate($id: ID!, $input: PromotionTranslationInput!,\n$languageCode: LanguageCodeEnum!) {\n  promotionTranslate(id: $id, input: $input, languageCode: $languageCode) {\n    errors {\n      field\n      code\n      message\n    }\n    promotion {\n      id\n      translation(languageCode: $languageCode) {\n        id\n        language {\n          code\n        }\n        name\n        description\n      }\n    }\n  }\n}\n'

def translate_promotion(staff_api_client, promotion_id, language_code='EN', input=None):
    if False:
        i = 10
        return i + 15
    variables = {'id': promotion_id, 'languageCode': language_code, 'input': input}
    response = staff_api_client.post_graphql(PROMOTION_TRANSLATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['promotionTranslate']['errors'] == []
    data = content['data']['promotionTranslate']['promotion']['translation']
    assert data['id'] is not None
    return data