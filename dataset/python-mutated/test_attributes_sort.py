import graphene
from .....attribute import AttributeType
from .....attribute.models import AssignedProductAttributeValue, Attribute
from .....attribute.utils import associate_attribute_values_to_instance
from ....tests.utils import get_graphql_content
ATTRIBUTES_SORT_QUERY = '\n    query($sortBy: AttributeSortingInput) {\n      attributes(first: 10, sortBy: $sortBy) {\n        edges {\n          node {\n            slug\n          }\n        }\n      }\n    }\n'

def test_sort_attributes_by_slug(api_client):
    if False:
        print('Hello World!')
    Attribute.objects.bulk_create([Attribute(name='MyAttribute', slug='b', type=AttributeType.PRODUCT_TYPE), Attribute(name='MyAttribute', slug='a', type=AttributeType.PRODUCT_TYPE)])
    variables = {'sortBy': {'field': 'SLUG', 'direction': 'ASC'}}
    attributes = get_graphql_content(api_client.post_graphql(ATTRIBUTES_SORT_QUERY, variables))['data']['attributes']['edges']
    assert len(attributes) == 2
    assert attributes[0]['node']['slug'] == 'a'
    assert attributes[1]['node']['slug'] == 'b'

def test_sort_attributes_by_default_sorting(api_client):
    if False:
        return 10
    "Don't provide any sorting, this should sort by slug by default."
    Attribute.objects.bulk_create([Attribute(name='A', slug='b', type=AttributeType.PRODUCT_TYPE), Attribute(name='B', slug='a', type=AttributeType.PRODUCT_TYPE)])
    attributes = get_graphql_content(api_client.post_graphql(ATTRIBUTES_SORT_QUERY, {}))['data']['attributes']['edges']
    assert len(attributes) == 2
    assert attributes[0]['node']['slug'] == 'a'
    assert attributes[1]['node']['slug'] == 'b'

def test_attributes_of_products_are_sorted_on_variant(user_api_client, product, color_attribute, channel_USD):
    if False:
        while True:
            i = 10
    'Ensures the attributes of products and variants are sorted.'
    variant = product.variants.first()
    query = '\n        query($id: ID!, $channel: String) {\n            productVariant(id: $id, channel: $channel) {\n            attributes {\n                attribute {\n                id\n                }\n            }\n            }\n        }\n    '
    other_attribute = Attribute.objects.create(name='Other', slug='other')
    product.product_type.variant_attributes.set([color_attribute, other_attribute])
    m2m_rel_other_attr = other_attribute.attributevariant.last()
    m2m_rel_other_attr.sort_order = 0
    m2m_rel_other_attr.save(update_fields=['sort_order'])
    associate_attribute_values_to_instance(variant, color_attribute, color_attribute.values.first())
    expected_order = [other_attribute.pk, color_attribute.pk]
    node_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
    data = get_graphql_content(user_api_client.post_graphql(query, {'id': node_id, 'channel': channel_USD.slug}))['data']
    attributes = data['productVariant']['attributes']
    actual_order = [int(graphene.Node.from_global_id(attr['attribute']['id'])[1]) for attr in attributes]
    assert actual_order == expected_order

def test_attributes_of_products_are_sorted_on_product(user_api_client, product, color_attribute, channel_USD):
    if False:
        i = 10
        return i + 15
    'Ensures the attributes of products and variants are sorted.'
    query = '\n        query($id: ID!, $channel: String) {\n            product(id: $id, channel: $channel) {\n            attributes {\n                attribute {\n                id\n                }\n            }\n            }\n        }\n    '
    other_attribute = Attribute.objects.create(name='Other', slug='other')
    product.product_type.product_attributes.set([color_attribute, other_attribute])
    m2m_rel_other_attr = other_attribute.attributeproduct.last()
    m2m_rel_other_attr.sort_order = 0
    m2m_rel_other_attr.save(update_fields=['sort_order'])
    AssignedProductAttributeValue.objects.filter(product_id=product.pk).delete()
    associate_attribute_values_to_instance(product, color_attribute, color_attribute.values.first())
    expected_order = [other_attribute.pk, color_attribute.pk]
    node_id = graphene.Node.to_global_id('Product', product.pk)
    data = get_graphql_content(user_api_client.post_graphql(query, {'id': node_id, 'channel': channel_USD.slug}))['data']
    attributes = data['product']['attributes']
    actual_order = [int(graphene.Node.from_global_id(attr['attribute']['id'])[1]) for attr in attributes]
    assert actual_order == expected_order
ATTRIBUTE_CHOICES_SORT_QUERY = '\nquery($sortBy: AttributeChoicesSortingInput) {\n    attributes(first: 10) {\n        edges {\n            node {\n                slug\n                choices(first: 10, sortBy: $sortBy) {\n                    edges {\n                        node {\n                            name\n                            slug\n                        }\n                    }\n                }\n            }\n        }\n    }\n}\n'

def test_sort_attribute_choices_by_slug(api_client, attribute_choices_for_sorting):
    if False:
        return 10
    variables = {'sortBy': {'field': 'SLUG', 'direction': 'ASC'}}
    attributes = get_graphql_content(api_client.post_graphql(ATTRIBUTE_CHOICES_SORT_QUERY, variables))['data']['attributes']
    choices = attributes['edges'][0]['node']['choices']['edges']
    assert len(choices) == 3
    assert choices[0]['node']['slug'] == 'absorb'
    assert choices[1]['node']['slug'] == 'summer'
    assert choices[2]['node']['slug'] == 'zet'

def test_sort_attribute_choices_by_name(api_client, attribute_choices_for_sorting):
    if False:
        i = 10
        return i + 15
    variables = {'sortBy': {'field': 'NAME', 'direction': 'ASC'}}
    attributes = get_graphql_content(api_client.post_graphql(ATTRIBUTE_CHOICES_SORT_QUERY, variables))['data']['attributes']
    choices = attributes['edges'][0]['node']['choices']['edges']
    assert len(choices) == 3
    assert choices[0]['node']['name'] == 'Apex'
    assert choices[1]['node']['name'] == 'Global'
    assert choices[2]['node']['name'] == 'Police'