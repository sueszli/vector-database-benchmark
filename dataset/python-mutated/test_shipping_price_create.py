import json
from unittest import mock
import graphene
import pytest
from django.utils.functional import SimpleLazyObject
from freezegun import freeze_time
from .....core.utils.json_serializer import CustomJsonEncoder
from .....shipping.error_codes import ShippingErrorCode
from .....shipping.models import ShippingMethod
from .....tests.utils import dummy_editorjs
from .....webhook.event_types import WebhookEventAsyncType
from .....webhook.payloads import generate_meta, generate_requestor
from ....core.enums import WeightUnitsEnum
from ....tests.utils import get_graphql_content
from ...types import PostalCodeRuleInclusionTypeEnum, ShippingMethodTypeEnum
PRICE_BASED_SHIPPING_MUTATION = '\n    mutation createShippingPrice(\n        $type: ShippingMethodTypeEnum,\n        $name: String!,\n        $description: JSONString,\n        $shippingZone: ID!,\n        $maximumDeliveryDays: Int,\n        $minimumDeliveryDays: Int,\n        $addPostalCodeRules: [ShippingPostalCodeRulesCreateInputRange!]\n        $deletePostalCodeRules: [ID!]\n        $inclusionType: PostalCodeRuleInclusionTypeEnum\n        $taxClass: ID\n    ) {\n        shippingPriceCreate(\n            input: {\n                name: $name, shippingZone: $shippingZone, type: $type,\n                maximumDeliveryDays: $maximumDeliveryDays,\n                minimumDeliveryDays: $minimumDeliveryDays,\n                addPostalCodeRules: $addPostalCodeRules,\n                deletePostalCodeRules: $deletePostalCodeRules,\n                inclusionType: $inclusionType,\n                description: $description,\n                taxClass: $taxClass\n            }) {\n            errors {\n                field\n                code\n            }\n            shippingZone {\n                id\n            }\n            shippingMethod {\n                id\n                name\n                description\n                channelListings {\n                    price {\n                        amount\n                    }\n                    minimumOrderPrice {\n                        amount\n                    }\n                    maximumOrderPrice {\n                        amount\n                    }\n                }\n                taxClass {\n                    id\n                }\n                type\n                minimumDeliveryDays\n                maximumDeliveryDays\n                postalCodeRules {\n                    start\n                    end\n                }\n            }\n        }\n    }\n'

@pytest.mark.parametrize('postal_code_rules', [[{'start': 'HB3', 'end': 'HB6'}], []])
def test_create_shipping_method(staff_api_client, shipping_zone, postal_code_rules, permission_manage_shipping, tax_classes):
    if False:
        i = 10
        return i + 15
    name = 'DHL'
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    max_del_days = 10
    min_del_days = 3
    description = dummy_editorjs('description', True)
    tax_class_id = graphene.Node.to_global_id('TaxClass', tax_classes[0].pk)
    variables = {'shippingZone': shipping_zone_id, 'name': name, 'description': description, 'type': ShippingMethodTypeEnum.PRICE.name, 'maximumDeliveryDays': max_del_days, 'minimumDeliveryDays': min_del_days, 'addPostalCodeRules': postal_code_rules, 'deletePostalCodeRules': [], 'inclusionType': PostalCodeRuleInclusionTypeEnum.EXCLUDE.name, 'taxClass': tax_class_id}
    response = staff_api_client.post_graphql(PRICE_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    errors = data['errors']
    assert not errors
    assert data['shippingMethod']['name'] == name
    assert data['shippingMethod']['description'] == description
    assert data['shippingMethod']['type'] == ShippingMethodTypeEnum.PRICE.name
    assert data['shippingZone']['id'] == shipping_zone_id
    assert data['shippingMethod']['minimumDeliveryDays'] == min_del_days
    assert data['shippingMethod']['maximumDeliveryDays'] == max_del_days
    assert data['shippingMethod']['postalCodeRules'] == postal_code_rules
    assert data['shippingMethod']['taxClass']['id'] == tax_class_id

@freeze_time('2022-05-12 12:00:00')
@mock.patch('saleor.plugins.webhook.plugin.get_webhooks_for_event')
@mock.patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
def test_create_shipping_method_trigger_webhook(mocked_webhook_trigger, mocked_get_webhooks_for_event, any_webhook, staff_api_client, shipping_zone, permission_manage_shipping, settings):
    if False:
        i = 10
        return i + 15
    mocked_get_webhooks_for_event.return_value = [any_webhook]
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    name = 'DHL'
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    max_del_days = 10
    min_del_days = 3
    description = dummy_editorjs('description', True)
    variables = {'shippingZone': shipping_zone_id, 'name': name, 'description': description, 'type': ShippingMethodTypeEnum.PRICE.name, 'maximumDeliveryDays': max_del_days, 'minimumDeliveryDays': min_del_days, 'addPostalCodeRules': [{'start': 'HB3', 'end': 'HB6'}], 'deletePostalCodeRules': [], 'inclusionType': PostalCodeRuleInclusionTypeEnum.EXCLUDE.name}
    response = staff_api_client.post_graphql(PRICE_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    shipping_method = ShippingMethod.objects.last()
    errors = data['errors']
    assert not errors
    assert shipping_method
    mocked_webhook_trigger.assert_called_once_with(json.dumps({'id': graphene.Node.to_global_id('ShippingMethodType', shipping_method.id), 'meta': generate_meta(requestor_data=generate_requestor(SimpleLazyObject(lambda : staff_api_client.user)))}, cls=CustomJsonEncoder), WebhookEventAsyncType.SHIPPING_PRICE_CREATED, [any_webhook], shipping_method, SimpleLazyObject(lambda : staff_api_client.user))

def test_create_shipping_method_minimum_delivery_days_higher_than_maximum(staff_api_client, shipping_zone, permission_manage_shipping):
    if False:
        return 10
    name = 'DHL'
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    max_del_days = 3
    min_del_days = 10
    variables = {'shippingZone': shipping_zone_id, 'name': name, 'type': ShippingMethodTypeEnum.PRICE.name, 'maximumDeliveryDays': max_del_days, 'minimumDeliveryDays': min_del_days}
    response = staff_api_client.post_graphql(PRICE_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    errors = data['errors']
    assert not data['shippingMethod']
    assert len(errors) == 1
    assert errors[0]['code'] == ShippingErrorCode.INVALID.name
    assert errors[0]['field'] == 'minimumDeliveryDays'

def test_create_shipping_method_minimum_delivery_days_below_0(staff_api_client, shipping_zone, permission_manage_shipping):
    if False:
        for i in range(10):
            print('nop')
    name = 'DHL'
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    max_del_days = 3
    min_del_days = -1
    variables = {'shippingZone': shipping_zone_id, 'name': name, 'type': ShippingMethodTypeEnum.PRICE.name, 'maximumDeliveryDays': max_del_days, 'minimumDeliveryDays': min_del_days}
    response = staff_api_client.post_graphql(PRICE_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    errors = data['errors']
    assert not data['shippingMethod']
    assert len(errors) == 1
    assert errors[0]['code'] == ShippingErrorCode.INVALID.name
    assert errors[0]['field'] == 'minimumDeliveryDays'

def test_create_shipping_method_maximum_delivery_days_below_0(staff_api_client, shipping_zone, permission_manage_shipping):
    if False:
        i = 10
        return i + 15
    name = 'DHL'
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    max_del_days = -1
    min_del_days = 10
    variables = {'shippingZone': shipping_zone_id, 'name': name, 'type': ShippingMethodTypeEnum.PRICE.name, 'maximumDeliveryDays': max_del_days, 'minimumDeliveryDays': min_del_days}
    response = staff_api_client.post_graphql(PRICE_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    errors = data['errors']
    assert not data['shippingMethod']
    assert len(errors) == 1
    assert errors[0]['code'] == ShippingErrorCode.INVALID.name
    assert errors[0]['field'] == 'maximumDeliveryDays'

def test_create_shipping_method_postal_code_duplicate_entry(staff_api_client, shipping_zone, permission_manage_shipping):
    if False:
        for i in range(10):
            print('nop')
    name = 'DHL'
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    max_del_days = 10
    min_del_days = 3
    postal_code_rules = [{'start': 'HB3', 'end': 'HB6'}, {'start': 'HB3', 'end': 'HB6'}]
    variables = {'shippingZone': shipping_zone_id, 'name': name, 'type': ShippingMethodTypeEnum.PRICE.name, 'maximumDeliveryDays': max_del_days, 'minimumDeliveryDays': min_del_days, 'addPostalCodeRules': postal_code_rules, 'inclusionType': PostalCodeRuleInclusionTypeEnum.EXCLUDE.name}
    response = staff_api_client.post_graphql(PRICE_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    errors = data['errors']
    assert not data['shippingMethod']
    assert len(errors) == 1
    assert errors[0]['code'] == ShippingErrorCode.ALREADY_EXISTS.name
    assert errors[0]['field'] == 'addPostalCodeRules'

def test_create_shipping_method_postal_code_missing_inclusion_type(staff_api_client, shipping_zone, permission_manage_shipping):
    if False:
        return 10
    name = 'DHL'
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    max_del_days = 10
    min_del_days = 3
    postal_code_rules = [{'start': 'HB3', 'end': 'HB6'}]
    variables = {'shippingZone': shipping_zone_id, 'name': name, 'type': ShippingMethodTypeEnum.PRICE.name, 'maximumDeliveryDays': max_del_days, 'minimumDeliveryDays': min_del_days, 'addPostalCodeRules': postal_code_rules}
    response = staff_api_client.post_graphql(PRICE_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    errors = data['errors']
    assert not data['shippingMethod']
    assert len(errors) == 1
    assert errors[0]['code'] == ShippingErrorCode.REQUIRED.name
    assert errors[0]['field'] == 'inclusionType'
WEIGHT_BASED_SHIPPING_MUTATION = '\n    mutation createShippingPrice(\n        $type: ShippingMethodTypeEnum\n        $name: String!\n        $shippingZone: ID!\n        $maximumOrderWeight: WeightScalar\n        $minimumOrderWeight: WeightScalar\n        ) {\n        shippingPriceCreate(\n            input: {\n                name: $name,shippingZone: $shippingZone,\n                minimumOrderWeight:$minimumOrderWeight,\n                maximumOrderWeight: $maximumOrderWeight,\n                type: $type\n            }) {\n            errors {\n                field\n                code\n            }\n            shippingMethod {\n                minimumOrderWeight {\n                    value\n                    unit\n                }\n                maximumOrderWeight {\n                    value\n                    unit\n                }\n            }\n            shippingZone {\n                id\n            }\n        }\n    }\n'

@pytest.mark.parametrize(('min_weight', 'max_weight', 'expected_min_weight', 'expected_max_weight'), [(10.32, 15.64, {'value': 10.32, 'unit': WeightUnitsEnum.KG.name}, {'value': 15.64, 'unit': WeightUnitsEnum.KG.name}), (10.92, None, {'value': 10.92, 'unit': WeightUnitsEnum.KG.name}, None)])
def test_create_weight_based_shipping_method(shipping_zone, staff_api_client, min_weight, max_weight, expected_min_weight, expected_max_weight, permission_manage_shipping):
    if False:
        while True:
            i = 10
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'shippingZone': shipping_zone_id, 'name': 'DHL', 'minimumOrderWeight': min_weight, 'maximumOrderWeight': max_weight, 'type': ShippingMethodTypeEnum.WEIGHT.name}
    response = staff_api_client.post_graphql(WEIGHT_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    assert data['shippingMethod']['minimumOrderWeight'] == expected_min_weight
    assert data['shippingMethod']['maximumOrderWeight'] == expected_max_weight
    assert data['shippingZone']['id'] == shipping_zone_id

def test_create_weight_shipping_method_errors(shipping_zone, staff_api_client, permission_manage_shipping):
    if False:
        return 10
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'shippingZone': shipping_zone_id, 'name': 'DHL', 'minimumOrderWeight': 20, 'maximumOrderWeight': 15, 'type': ShippingMethodTypeEnum.WEIGHT.name}
    response = staff_api_client.post_graphql(WEIGHT_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    assert data['errors'][0]['code'] == ShippingErrorCode.MAX_LESS_THAN_MIN.name

def test_create_shipping_method_with_negative_min_weight(shipping_zone, staff_api_client, permission_manage_shipping):
    if False:
        print('Hello World!')
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'shippingZone': shipping_zone_id, 'name': 'DHL', 'minimumOrderWeight': -20, 'type': ShippingMethodTypeEnum.WEIGHT.name}
    response = staff_api_client.post_graphql(WEIGHT_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    error = data['errors'][0]
    assert error['field'] == 'minimumOrderWeight'
    assert error['code'] == ShippingErrorCode.INVALID.name

def test_create_shipping_method_with_negative_max_weight(shipping_zone, staff_api_client, permission_manage_shipping):
    if False:
        while True:
            i = 10
    shipping_zone_id = graphene.Node.to_global_id('ShippingZone', shipping_zone.pk)
    variables = {'shippingZone': shipping_zone_id, 'name': 'DHL', 'maximumOrderWeight': -15, 'type': ShippingMethodTypeEnum.WEIGHT.name}
    response = staff_api_client.post_graphql(WEIGHT_BASED_SHIPPING_MUTATION, variables, permissions=[permission_manage_shipping])
    content = get_graphql_content(response)
    data = content['data']['shippingPriceCreate']
    error = data['errors'][0]
    assert error['field'] == 'maximumOrderWeight'
    assert error['code'] == ShippingErrorCode.INVALID.name