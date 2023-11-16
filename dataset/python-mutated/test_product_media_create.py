import json
import os
from unittest.mock import patch
import graphene
import pytest
from .....graphql.tests.utils import get_graphql_content, get_multipart_request_body
from .....product import ProductMediaTypes
from .....product.error_codes import ProductErrorCode
from .....product.tests.utils import create_image, create_zip_file_with_image_ext
PRODUCT_MEDIA_CREATE_QUERY = '\n    mutation createProductMedia(\n        $product: ID!,\n        $image: Upload,\n        $mediaUrl: String,\n        $alt: String\n    ) {\n        productMediaCreate(input: {\n            product: $product,\n            mediaUrl: $mediaUrl,\n            alt: $alt,\n            image: $image\n        }) {\n            product {\n                media {\n                    url\n                    alt\n                    type\n                    oembedData\n                }\n            }\n            errors {\n                code\n                field\n            }\n        }\n    }\n'

@patch('saleor.plugins.manager.PluginsManager.product_media_created')
@patch('saleor.plugins.manager.PluginsManager.product_updated')
def test_product_media_create_mutation(product_updated_mock, product_media_created, monkeypatch, staff_api_client, product, permission_manage_products, media_root):
    if False:
        print('Hello World!')
    staff_api_client.user.user_permissions.add(permission_manage_products)
    (image_file, image_name) = create_image()
    variables = {'product': graphene.Node.to_global_id('Product', product.id), 'alt': '', 'image': image_name}
    body = get_multipart_request_body(PRODUCT_MEDIA_CREATE_QUERY, variables, image_file, image_name)
    response = staff_api_client.post_multipart(body)
    get_graphql_content(response)
    product.refresh_from_db()
    product_image = product.media.last()
    assert product_image.image.file
    (img_name, format) = os.path.splitext(image_file._name)
    file_name = product_image.image.name
    assert file_name != image_file._name
    assert file_name.startswith(f'products/{img_name}')
    assert file_name.endswith(format)
    product_updated_mock.assert_called_once_with(product)
    product_media_created.assert_called_once_with(product_image)

def test_product_media_create_mutation_without_file(monkeypatch, staff_api_client, product, permission_manage_products, media_root):
    if False:
        i = 10
        return i + 15
    variables = {'product': graphene.Node.to_global_id('Product', product.id), 'image': 'image name'}
    body = get_multipart_request_body(PRODUCT_MEDIA_CREATE_QUERY, variables, file='', file_name='name')
    response = staff_api_client.post_multipart(body, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    errors = content['data']['productMediaCreate']['errors']
    assert errors[0]['field'] == 'image'
    assert errors[0]['code'] == ProductErrorCode.REQUIRED.name

@pytest.mark.vcr
def test_product_media_create_mutation_with_media_url(monkeypatch, staff_api_client, product, permission_manage_products, media_root):
    if False:
        for i in range(10):
            print('nop')
    variables = {'product': graphene.Node.to_global_id('Product', product.id), 'mediaUrl': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'alt': ''}
    body = get_multipart_request_body(PRODUCT_MEDIA_CREATE_QUERY, variables, file='', file_name='name')
    response = staff_api_client.post_multipart(body, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    media = content['data']['productMediaCreate']['product']['media']
    alt = 'Rick Astley - Never Gonna Give You Up (Official Music Video)'
    assert len(media) == 1
    assert media[0]['url'] == 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    assert media[0]['alt'] == alt
    assert media[0]['type'] == ProductMediaTypes.VIDEO
    oembed_data = json.loads(media[0]['oembedData'])
    assert oembed_data['url'] == 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    assert oembed_data['type'] == 'video'
    assert oembed_data['html'] is not None
    assert oembed_data['thumbnail_url'] == 'https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg'

def test_product_media_create_mutation_without_url_or_image(monkeypatch, staff_api_client, product, permission_manage_products, media_root):
    if False:
        print('Hello World!')
    variables = {'product': graphene.Node.to_global_id('Product', product.id), 'alt': 'Test Alt Text'}
    body = get_multipart_request_body(PRODUCT_MEDIA_CREATE_QUERY, variables, file='', file_name='name')
    response = staff_api_client.post_multipart(body, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    errors = content['data']['productMediaCreate']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == ProductErrorCode.REQUIRED.name
    assert errors[0]['field'] == 'input'

def test_product_media_create_mutation_with_both_url_and_image(monkeypatch, staff_api_client, product, permission_manage_products, media_root):
    if False:
        print('Hello World!')
    (image_file, image_name) = create_image()
    variables = {'product': graphene.Node.to_global_id('Product', product.id), 'mediaUrl': 'https://www.youtube.com/watch?v=SomeVideoID&ab_channel=Test', 'image': image_name, 'alt': 'Test Alt Text'}
    body = get_multipart_request_body(PRODUCT_MEDIA_CREATE_QUERY, variables, image_file, image_name)
    response = staff_api_client.post_multipart(body, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    errors = content['data']['productMediaCreate']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == ProductErrorCode.DUPLICATED_INPUT_ITEM.name
    assert errors[0]['field'] == 'input'

def test_product_media_create_mutation_with_unknown_url(monkeypatch, staff_api_client, product, permission_manage_products, media_root):
    if False:
        for i in range(10):
            print('nop')
    variables = {'product': graphene.Node.to_global_id('Product', product.id), 'mediaUrl': 'https://www.videohosting.com/SomeVideoID', 'alt': 'Test Alt Text'}
    body = get_multipart_request_body(PRODUCT_MEDIA_CREATE_QUERY, variables, file='', file_name='name')
    response = staff_api_client.post_multipart(body, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    errors = content['data']['productMediaCreate']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == ProductErrorCode.UNSUPPORTED_MEDIA_PROVIDER.name
    assert errors[0]['field'] == 'mediaUrl'

def test_invalid_product_media_create_mutation(staff_api_client, product, permission_manage_products):
    if False:
        for i in range(10):
            print('nop')
    query = '\n    mutation createProductMedia($image: Upload!, $product: ID!) {\n        productMediaCreate(input: {image: $image, product: $product}) {\n            media {\n                id\n                url\n                sortOrder\n            }\n            errors {\n                field\n                message\n            }\n        }\n    }\n    '
    (image_file, image_name) = create_zip_file_with_image_ext()
    variables = {'product': graphene.Node.to_global_id('Product', product.id), 'image': image_name}
    body = get_multipart_request_body(query, variables, image_file, image_name)
    response = staff_api_client.post_multipart(body, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    assert content['data']['productMediaCreate']['errors'] == [{'field': 'image', 'message': 'Invalid file type.'}]
    product.refresh_from_db()
    assert product.media.count() == 0