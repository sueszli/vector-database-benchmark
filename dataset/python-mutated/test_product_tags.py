from unittest.mock import MagicMock
import graphene
from django.core.files import File
from django.templatetags.static import static
from ...thumbnail.models import Thumbnail
from ..product_images import get_product_image_placeholder, get_product_image_thumbnail_url

def test_get_product_image_thumbnail_no_instance():
    if False:
        i = 10
        return i + 15
    output = get_product_image_thumbnail_url(product_media=None, size=10)
    assert output == static('images/placeholder32.png')

def test_get_product_image_thumbnail_no_media_image(product_with_image):
    if False:
        for i in range(10):
            print('nop')
    media = product_with_image.media.first()
    media.image = None
    media.save(update_fields=['image'])
    output = get_product_image_thumbnail_url(product_media=None, size=10)
    assert output == static('images/placeholder32.png')

def test_get_product_image_thumbnail_proxy_url_returned(product_with_image):
    if False:
        print('Hello World!')
    media = product_with_image.media.first()
    output = get_product_image_thumbnail_url(product_media=media, size=10)
    assert output == f"/thumbnail/{graphene.Node.to_global_id('ProductMedia', media.pk)}/32/"

def test_get_product_image_thumbnail_url_returned(product_with_image):
    if False:
        return 10
    media = product_with_image.media.first()
    size = 128
    thumbnail_mock = MagicMock(spec=File)
    thumbnail_mock.name = 'thumbnail_image.jpg'
    thumbnail = Thumbnail.objects.create(product_media=media, image=thumbnail_mock, size=size)
    output = get_product_image_thumbnail_url(product_media=media, size=size)
    assert output == thumbnail.image.url

def test_choose_placeholder(settings):
    if False:
        return 10
    settings.PLACEHOLDER_IMAGES = {32: '20_placeholder', 64: '32_placeholder', 4096: '10_placeholder'}
    assert get_product_image_placeholder(64) == static(settings.PLACEHOLDER_IMAGES[64])
    assert get_product_image_placeholder(15) == static(settings.PLACEHOLDER_IMAGES[32])
    assert get_product_image_placeholder(5000) == static(settings.PLACEHOLDER_IMAGES[4096])