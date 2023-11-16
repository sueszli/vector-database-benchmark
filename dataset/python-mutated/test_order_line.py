from decimal import Decimal
from unittest.mock import MagicMock, patch
import graphene
from django.core.files import File
from prices import Money, TaxedMoney
from .....core.prices import quantize_price
from .....order import OrderStatus
from .....order.interface import OrderTaxedPricesData
from .....thumbnail.models import Thumbnail
from .....warehouse.models import Stock
from ....core.enums import ThumbnailFormatEnum
from ....tests.utils import get_graphql_content

def test_order_line_query(staff_api_client, permission_group_manage_orders, fulfilled_order):
    if False:
        print('Hello World!')
    order = fulfilled_order
    query = '\n        query OrdersQuery {\n            orders(first: 1) {\n                edges {\n                    node {\n                        lines {\n                            thumbnail(size: 540) {\n                                url\n                            }\n                            variant {\n                                id\n                            }\n                            quantity\n                            allocations {\n                                id\n                                quantity\n                                warehouse {\n                                    id\n                                }\n                            }\n                            unitPrice {\n                                currency\n                                gross {\n                                    amount\n                                }\n                            }\n                            totalPrice {\n                                currency\n                                gross {\n                                    amount\n                                }\n                            }\n                            undiscountedTotalPrice {\n                                currency\n                                gross {\n                                    amount\n                                }\n                            }\n                            metadata {\n                                key\n                                value\n                            }\n                            privateMetadata {\n                                key\n                                value\n                            }\n                            taxClass {\n                                name\n                            }\n                            taxClassName\n                            taxClassMetadata {\n                                key\n                                value\n                            }\n                            taxClassPrivateMetadata {\n                                key\n                                value\n                            }\n                            taxRate\n                        }\n                    }\n                }\n            }\n        }\n    '
    line = order.lines.first()
    metadata_key = 'md key'
    metadata_value = 'md value'
    line.store_value_in_private_metadata({metadata_key: metadata_value})
    line.store_value_in_metadata({metadata_key: metadata_value})
    line.save()
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(query)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    first_order_data_line = order_data['lines'][0]
    variant_id = graphene.Node.to_global_id('ProductVariant', line.variant.pk)
    assert first_order_data_line['thumbnail'] is None
    assert first_order_data_line['variant']['id'] == variant_id
    assert first_order_data_line['quantity'] == line.quantity
    assert first_order_data_line['unitPrice']['currency'] == line.unit_price.currency
    assert first_order_data_line['metadata'] == [{'key': metadata_key, 'value': metadata_value}]
    assert first_order_data_line['privateMetadata'] == [{'key': metadata_key, 'value': metadata_value}]
    expected_unit_price = Money(amount=str(first_order_data_line['unitPrice']['gross']['amount']), currency='USD')
    assert first_order_data_line['totalPrice']['currency'] == line.unit_price.currency
    assert first_order_data_line['undiscountedTotalPrice']['currency'] == line.currency
    assert expected_unit_price == line.unit_price.gross
    expected_total_price = Money(amount=str(first_order_data_line['totalPrice']['gross']['amount']), currency='USD')
    assert expected_total_price == line.unit_price.gross * line.quantity
    expected_undiscounted_total_price = Money(amount=str(first_order_data_line['undiscountedTotalPrice']['gross']['amount']), currency='USD')
    assert expected_undiscounted_total_price == line.undiscounted_total_price.gross
    allocation = line.allocations.first()
    allocation_id = graphene.Node.to_global_id('Allocation', allocation.pk)
    warehouse_id = graphene.Node.to_global_id('Warehouse', allocation.stock.warehouse.pk)
    assert first_order_data_line['allocations'] == [{'id': allocation_id, 'quantity': allocation.quantity_allocated, 'warehouse': {'id': warehouse_id}}]
    line_tax_class = line.variant.product.tax_class
    assert first_order_data_line['taxClass']['name'] == line_tax_class.name
    assert first_order_data_line['taxClassName'] == line_tax_class.name
    assert first_order_data_line['taxClassMetadata'][0]['key'] == list(line_tax_class.metadata.keys())[0]
    assert first_order_data_line['taxClassMetadata'][0]['value'] == list(line_tax_class.metadata.values())[0]
    assert first_order_data_line['taxClassPrivateMetadata'][0]['key'] == list(line_tax_class.private_metadata.keys())[0]
    assert first_order_data_line['taxClassPrivateMetadata'][0]['value'] == list(line_tax_class.private_metadata.values())[0]

def test_denormalized_tax_class_in_orderline_query(staff_api_client, permission_group_manage_orders, fulfilled_order):
    if False:
        print('Hello World!')
    order = fulfilled_order
    query = '\n            query OrdersQuery {\n                orders(first: 1) {\n                    edges {\n                        node {\n                            lines {\n                                thumbnail(size: 540) {\n                                    url\n                                }\n                                variant {\n                                    id\n                                }\n                                quantity\n                                allocations {\n                                    id\n                                    quantity\n                                    warehouse {\n                                        id\n                                    }\n                                }\n                                unitPrice {\n                                    currency\n                                    gross {\n                                        amount\n                                    }\n                                }\n                                totalPrice {\n                                    currency\n                                    gross {\n                                        amount\n                                    }\n                                }\n                                metadata {\n                                    key\n                                    value\n                                }\n                                privateMetadata {\n                                    key\n                                    value\n                                }\n                                taxClass {\n                                    name\n                                }\n                                taxClassName\n                                taxClassMetadata {\n                                    key\n                                    value\n                                }\n                                taxClassPrivateMetadata {\n                                    key\n                                    value\n                                }\n                                taxRate\n                            }\n                        }\n                    }\n                }\n            }\n        '
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    line_tax_class = order.lines.first().tax_class
    assert line_tax_class
    line_tax_class.delete()
    response = staff_api_client.post_graphql(query)
    content = get_graphql_content(response)
    line_data = content['data']['orders']['edges'][0]['node']['lines'][0]
    assert line_data['taxClass'] is None
    assert line_data['taxClassName'] == line_tax_class.name
    assert line_data['taxClassMetadata'][0]['key'] == list(line_tax_class.metadata.keys())[0]
    assert line_data['taxClassMetadata'][0]['value'] == list(line_tax_class.metadata.values())[0]
    assert line_data['taxClassPrivateMetadata'][0]['key'] == list(line_tax_class.private_metadata.keys())[0]
    assert line_data['taxClassPrivateMetadata'][0]['value'] == list(line_tax_class.private_metadata.values())[0]

def test_order_line_with_allocations(staff_api_client, permission_group_manage_orders, order_with_lines):
    if False:
        for i in range(10):
            print('nop')
    order = order_with_lines
    query = '\n        query OrdersQuery {\n            orders(first: 1) {\n                edges {\n                    node {\n                        lines {\n                            id\n                            allocations {\n                                id\n                                quantity\n                                warehouse {\n                                    id\n                                }\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    '
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(query)
    content = get_graphql_content(response)
    lines = content['data']['orders']['edges'][0]['node']['lines']
    for line in lines:
        (_, _id) = graphene.Node.from_global_id(line['id'])
        order_line = order.lines.get(pk=_id)
        allocations_from_query = {allocation['quantity'] for allocation in line['allocations']}
        allocations_from_db = set(order_line.allocations.values_list('quantity_allocated', flat=True))
        assert allocations_from_query == allocations_from_db
QUERY_ORDER_LINE_STOCKS = '\nquery OrderQuery($id: ID!) {\n    order(id: $id) {\n        number\n        lines {\n            id\n            quantity\n            quantityFulfilled\n            variant {\n                id\n                name\n                sku\n                stocks {\n                    warehouse {\n                        id\n                        name\n                    }\n                }\n            }\n        }\n    }\n}\n'

def test_query_order_line_stocks(staff_api_client, permission_group_manage_orders, order_with_lines_for_cc, warehouse, warehouse_for_cc):
    if False:
        return 10
    'Ensure that stocks for normal and click and collect warehouses are returned.'
    order = order_with_lines_for_cc
    variant = order.lines.first().variant
    variables = {'id': graphene.Node.to_global_id('Order', order.id)}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    Stock.objects.create(warehouse=warehouse, product_variant=variant, quantity=10)
    response = staff_api_client.post_graphql(QUERY_ORDER_LINE_STOCKS, variables)
    content = get_graphql_content(response)
    order_data = content['data']['order']
    assert order_data
    assert len(order_data['lines']) == 1
    assert {stock['warehouse']['name'] for stock in order_data['lines'][0]['variant']['stocks']} == {warehouse.name, warehouse_for_cc.name}
ORDERS_QUERY_LINE_THUMBNAIL = '\n    query OrdersQuery($size: Int, $format: ThumbnailFormatEnum) {\n        orders(first: 1) {\n            edges {\n                node {\n                    lines {\n                        id\n                        thumbnail(size: $size, format: $format) {\n                            url\n                        }\n                    }\n                }\n            }\n        }\n    }\n'

def test_order_query_no_thumbnail(staff_api_client, permission_group_manage_orders, order_line):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDERS_QUERY_LINE_THUMBNAIL)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    assert len(order_data['lines']) == 1
    assert not order_data['lines'][0]['thumbnail']

def test_order_query_product_image_size_and_format_given_proxy_url_returned(staff_api_client, permission_group_manage_orders, order_line, product_with_image, site_settings):
    if False:
        return 10
    order_line.variant.product = product_with_image
    media = product_with_image.media.first()
    format = ThumbnailFormatEnum.WEBP.name
    variables = {'size': 120, 'format': format}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDERS_QUERY_LINE_THUMBNAIL, variables)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    media_id = graphene.Node.to_global_id('ProductMedia', media.pk)
    domain = site_settings.site.domain
    assert len(order_data['lines']) == 1
    assert order_data['lines'][0]['thumbnail']['url'] == f'http://{domain}/thumbnail/{media_id}/128/{format.lower()}/'

def test_order_query_product_image_size_given_proxy_url_returned(staff_api_client, permission_group_manage_orders, order_line, product_with_image, site_settings):
    if False:
        return 10
    order_line.variant.product = product_with_image
    media = product_with_image.media.first()
    variables = {'size': 120}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDERS_QUERY_LINE_THUMBNAIL, variables)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    media_id = graphene.Node.to_global_id('ProductMedia', media.pk)
    assert len(order_data['lines']) == 1
    assert order_data['lines'][0]['thumbnail']['url'] == f'http://{site_settings.site.domain}/thumbnail/{media_id}/128/'

def test_order_query_product_image_size_given_thumbnail_url_returned(staff_api_client, permission_group_manage_orders, order_line, product_with_image, site_settings):
    if False:
        i = 10
        return i + 15
    order_line.variant.product = product_with_image
    media = product_with_image.media.first()
    thumbnail_mock = MagicMock(spec=File)
    thumbnail_mock.name = 'thumbnail_image.jpg'
    Thumbnail.objects.create(product_media=media, size=128, image=thumbnail_mock)
    variables = {'size': 120}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDERS_QUERY_LINE_THUMBNAIL, variables)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    assert len(order_data['lines']) == 1
    assert order_data['lines'][0]['thumbnail']['url'] == f'http://{site_settings.site.domain}/media/thumbnails/{thumbnail_mock.name}'

def test_order_query_variant_image_size_and_format_given_proxy_url_returned(staff_api_client, permission_group_manage_orders, order_line, variant_with_image, site_settings):
    if False:
        i = 10
        return i + 15
    order_line.variant = variant_with_image
    media = variant_with_image.media.first()
    format = ThumbnailFormatEnum.WEBP.name
    variables = {'size': 120, 'format': format}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDERS_QUERY_LINE_THUMBNAIL, variables)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    media_id = graphene.Node.to_global_id('ProductMedia', media.pk)
    domain = site_settings.site.domain
    assert len(order_data['lines']) == 1
    assert order_data['lines'][0]['thumbnail']['url'] == f'http://{domain}/thumbnail/{media_id}/128/{format.lower()}/'

def test_order_query_variant_image_size_given_proxy_url_returned(staff_api_client, permission_group_manage_orders, order_line, variant_with_image, site_settings):
    if False:
        return 10
    order_line.variant = variant_with_image
    media = variant_with_image.media.first()
    variables = {'size': 120}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDERS_QUERY_LINE_THUMBNAIL, variables)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    media_id = graphene.Node.to_global_id('ProductMedia', media.pk)
    assert len(order_data['lines']) == 1
    assert order_data['lines'][0]['thumbnail']['url'] == f'http://{site_settings.site.domain}/thumbnail/{media_id}/128/'

def test_order_query_variant_image_size_given_thumbnail_url_returned(staff_api_client, permission_group_manage_orders, order_line, variant_with_image, site_settings):
    if False:
        return 10
    order_line.variant = variant_with_image
    media = variant_with_image.media.first()
    thumbnail_mock = MagicMock(spec=File)
    thumbnail_mock.name = 'thumbnail_image.jpg'
    Thumbnail.objects.create(product_media=media, size=128, image=thumbnail_mock)
    variables = {'size': 120}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDERS_QUERY_LINE_THUMBNAIL, variables)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    assert len(order_data['lines']) == 1
    assert order_data['lines'][0]['thumbnail']['url'] == f'http://{site_settings.site.domain}/media/thumbnails/{thumbnail_mock.name}'
QUERY_LINE_TAX_CLASS_QUERY = '\n    query OrdersQuery {\n        orders(first: 1) {\n            edges {\n                node {\n                    lines {\n                        id\n                        taxClass {\n                            id\n                        }\n                    }\n                }\n            }\n        }\n    }\n'

def test_order_line_tax_class_query_by_staff(staff_api_client, permission_group_all_perms_all_channels, order_line):
    if False:
        i = 10
        return i + 15
    permission_group_all_perms_all_channels.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(QUERY_LINE_TAX_CLASS_QUERY)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    assert order_data['lines'][0]['taxClass']['id']

def test_order_line_tax_class_query_by_app(app_api_client, permission_manage_orders, order_line):
    if False:
        i = 10
        return i + 15
    app_api_client.app.permissions.add(permission_manage_orders)
    response = app_api_client.post_graphql(QUERY_LINE_TAX_CLASS_QUERY)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    assert order_data['lines'][0]['taxClass']['id']
UNDISCOUNTED_PRICE_QUERY = '\n        query OrdersQuery {\n            orders(first: 1) {\n                edges {\n                    node {\n                        lines {\n                            undiscountedUnitPrice {\n                                net {\n                                    amount\n                                }\n                                gross {\n                                    amount\n                                }\n                            }\n                        }\n                    }\n                }\n            }\n        }\n'

@patch('saleor.plugins.manager.PluginsManager.calculate_order_line_unit')
def test_order_query_undiscounted_prices_taxed(mocked_calculate_order_line_unit, staff_api_client, permission_group_all_perms_all_channels, fulfilled_order):
    if False:
        while True:
            i = 10
    order = fulfilled_order
    query = UNDISCOUNTED_PRICE_QUERY
    order.status = OrderStatus.UNCONFIRMED
    order.should_refresh_prices = True
    order.save(update_fields=['status', 'should_refresh_prices'])
    tc = order.channel.tax_configuration
    tc.prices_entered_with_tax = False
    tc.save(update_fields=['prices_entered_with_tax'])
    line = order.lines.first()
    tax_rate = line.tax_rate
    permission_group_all_perms_all_channels.user_set.add(staff_api_client.user)
    line_undiscounted_price = TaxedMoney(line.undiscounted_base_unit_price, line.undiscounted_base_unit_price * (1 + tax_rate))
    mocked_calculate_order_line_unit.return_value = OrderTaxedPricesData(undiscounted_price=line_undiscounted_price, price_with_discounts=line_undiscounted_price)
    line.undiscounted_unit_price_gross_amount = line.undiscounted_unit_price_net_amount
    line.save(update_fields=['undiscounted_unit_price_gross_amount'])
    response = staff_api_client.post_graphql(query)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    first_order_data_line_price = order_data['lines'][0]['undiscountedUnitPrice']
    assert first_order_data_line_price['net']['amount'] == line.unit_price.net.amount
    expected_gross = quantize_price(line.unit_price.net.amount * (tax_rate + 1), line.currency)
    result_gross = quantize_price(Decimal(first_order_data_line_price['gross']['amount']), line.currency)
    assert result_gross == expected_gross

def test_order_query_undiscounted_prices_no_tax(staff_api_client, permission_group_all_perms_all_channels, order_with_lines):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    query = UNDISCOUNTED_PRICE_QUERY
    order.status = OrderStatus.UNCONFIRMED
    order.should_refresh_prices = True
    order.save(update_fields=['status', 'should_refresh_prices'])
    tc = order.channel.tax_configuration
    tc.country_exceptions.all().delete()
    tc.prices_entered_with_tax = False
    tc.tax_calculation_strategy = None
    tc.charge_taxes = False
    tc.save(update_fields=['prices_entered_with_tax', 'tax_calculation_strategy', 'charge_taxes'])
    line = order.lines.first()
    line.undiscounted_unit_price_gross_amount = line.undiscounted_unit_price_net_amount
    line.tax_rate = Decimal(0)
    line.save()
    permission_group_all_perms_all_channels.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(query)
    content = get_graphql_content(response)
    order_data = content['data']['orders']['edges'][0]['node']
    first_order_data_line_price = order_data['lines'][0]['undiscountedUnitPrice']
    assert first_order_data_line_price['net']['amount'] == line.unit_price.net.amount
    assert first_order_data_line_price['gross']['amount'] == line.unit_price.net.amount