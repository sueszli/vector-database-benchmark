import json
import pytest
from ..schema.email import get_order_confirmation_markup, get_organization, get_product_data

def test_get_organization(site_settings):
    if False:
        i = 10
        return i + 15
    example_name = 'Saleor Brand Name'
    site = site_settings.site
    site.name = example_name
    site.save()
    result = get_organization()
    assert result['name'] == example_name

def test_get_product_data_without_image(order_with_lines):
    if False:
        while True:
            i = 10
    'Tested OrderLine Product has no image assigned.'
    line = order_with_lines.lines.first()
    organization = get_organization()
    result = get_product_data(line, organization)
    assert 'image' not in result['itemOffered']

def test_get_product_data_without_sku(order_with_lines):
    if False:
        print('Hello World!')
    'Tested OrderLine Product has no image assigned.'
    line = order_with_lines.lines.first()
    line.product_sku = None
    line.save()
    organization = get_organization()
    result = get_product_data(line, organization)
    assert 'image' not in result['itemOffered']

def test_get_product_data_with_image(order_with_lines, product_with_image):
    if False:
        while True:
            i = 10
    line = order_with_lines.lines.first()
    variant = product_with_image.variants.first()
    line.variant = variant
    line.product_name = str(variant.product)
    line.variant_name = str(variant)
    line.save()
    organization = get_organization()
    result = get_product_data(line, organization)
    assert 'image' in result['itemOffered']
    assert result['itemOffered']['name'] == variant.display_product()

def test_get_product_data_without_line_variant(order_with_lines):
    if False:
        i = 10
        return i + 15
    'Tested OrderLine Product has no image assigned.'
    line = order_with_lines.lines.first()
    organization = get_organization()
    line.variant = None
    line.save()
    assert not line.variant
    result = get_product_data(line, organization)
    assert result == {}

def test_get_order_confirmation_markup(order_with_lines):
    if False:
        for i in range(10):
            print('nop')
    try:
        result = get_order_confirmation_markup(order_with_lines)
    except TypeError:
        pytest.fail('Function output is not JSON serializable')
    try:
        json.loads(result)
    except ValueError:
        pytest.fail('Response is not a valid json')