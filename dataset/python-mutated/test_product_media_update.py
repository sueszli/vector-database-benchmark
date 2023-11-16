from unittest.mock import patch
import graphene
from .....graphql.tests.utils import get_graphql_content

@patch('saleor.plugins.manager.PluginsManager.product_media_updated')
@patch('saleor.plugins.manager.PluginsManager.product_updated')
def test_product_image_update_mutation(product_updated_mock, product_media_update_mock, monkeypatch, staff_api_client, product_with_image, permission_manage_products):
    if False:
        i = 10
        return i + 15
    query = '\n    mutation updateProductMedia($mediaId: ID!, $alt: String) {\n        productMediaUpdate(id: $mediaId, input: {alt: $alt}) {\n            media {\n                alt\n            }\n        }\n    }\n    '
    media_obj = product_with_image.media.first()
    alt = 'damage alt'
    assert media_obj.alt != alt
    variables = {'alt': alt, 'mediaId': graphene.Node.to_global_id('ProductMedia', media_obj.id)}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    media_obj.refresh_from_db()
    assert content['data']['productMediaUpdate']['media']['alt'] == alt
    assert media_obj.alt == alt
    product_updated_mock.assert_called_once_with(product_with_image)
    product_media_update_mock.assert_called_once_with(media_obj)