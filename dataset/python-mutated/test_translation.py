import pytest
from ...attribute.models import AttributeValueTranslation
from ...core.utils.translations import get_translation
from ...tests.utils import dummy_editorjs
from ..models import ProductTranslation, ProductVariantTranslation

@pytest.fixture
def product_translation_pl(product):
    if False:
        print('Hello World!')
    return ProductTranslation.objects.create(language_code='pl', product=product, name='Polish name', description=dummy_editorjs('Polish description.'))

@pytest.fixture
def attribute_value_translation_fr(translated_attribute):
    if False:
        while True:
            i = 10
    value = translated_attribute.attribute.values.first()
    return AttributeValueTranslation.objects.create(language_code='fr', attribute_value=value, name='French name')

def test_translation(product, settings, product_translation_fr):
    if False:
        for i in range(10):
            print('nop')
    assert get_translation(product).name == 'Test product'
    assert not get_translation(product).description
    settings.LANGUAGE_CODE = 'fr'
    assert get_translation(product).name == 'French name'
    assert get_translation(product).description == dummy_editorjs('French description.')

def test_translation_str_returns_str_of_instance(product, product_translation_fr, settings):
    if False:
        for i in range(10):
            print('nop')
    assert get_translation(product).name == str(product)
    settings.LANGUAGE_CODE = 'fr'
    assert str(get_translation(product).translation) == str(product_translation_fr)

def test_wrapper_gets_proper_wrapper(product, product_translation_fr, settings, product_translation_pl):
    if False:
        print('Hello World!')
    assert get_translation(product).translation is None
    settings.LANGUAGE_CODE = 'fr'
    assert get_translation(product).translation == product_translation_fr
    settings.LANGUAGE_CODE = 'pl'
    assert get_translation(product).translation == product_translation_pl

def test_getattr(product, settings, product_translation_fr, product_type):
    if False:
        print('Hello World!')
    settings.LANGUAGE_CODE = 'fr'
    assert get_translation(product).product_type == product_type

def test_translation_not_override_id(settings, product, product_translation_fr):
    if False:
        for i in range(10):
            print('nop')
    settings.LANGUAGE_CODE = 'fr'
    translated_product = get_translation(product)
    assert translated_product.id == product.id
    assert not translated_product.id == product_translation_fr

def test_product_variant_translation(settings, variant):
    if False:
        while True:
            i = 10
    settings.LANGUAGE_CODE = 'fr'
    french_name = 'French name'
    ProductVariantTranslation.objects.create(language_code='fr', name=french_name, product_variant=variant)
    assert get_translation(variant).name == french_name

def test_attribute_value_translation(settings, product, attribute_value_translation_fr):
    if False:
        print('Hello World!')
    attribute = product.product_type.product_attributes.first().values.first()
    assert not get_translation(attribute).name == 'French name'
    settings.LANGUAGE_CODE = 'fr'
    assert get_translation(attribute).name == 'French name'

def test_voucher_translation(settings, voucher, voucher_translation_fr):
    if False:
        print('Hello World!')
    assert not get_translation(voucher).name == 'French name'
    settings.LANGUAGE_CODE = 'fr'
    assert get_translation(voucher).name == 'French name'