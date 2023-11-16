from decimal import Decimal
from unittest.mock import patch
import graphene
import pytest
from graphene import Node
from .....checkout import calculations
from .....checkout.fetch import fetch_checkout_info, fetch_checkout_lines
from .....checkout.models import Checkout
from .....checkout.utils import add_variants_to_checkout, set_external_shipping_id
from .....plugins.manager import get_plugins_manager
from .....product.models import ProductVariant, ProductVariantChannelListing
from .....warehouse.models import Stock
from ....core.utils import to_global_id_or_none
from ....tests.utils import get_graphql_content
from ...mutations.utils import CheckoutLineData
CHECKOUT_GIFT_CARD_QUERY = '\n    query CheckoutGiftCard {\n      checkouts(first: 100) {\n        edges {\n          node {\n            id\n            giftCards {\n              id\n              isActive\n              code\n              last4CodeChars\n              currentBalance {\n                amount\n              }\n            }\n          }\n        }\n      }\n    }\n'
FRAGMENT_PRICE = '\n    fragment Price on TaxedMoney {\n      gross {\n        amount\n        currency\n      }\n      net {\n        amount\n        currency\n      }\n    }\n'
FRAGMENT_PRODUCT_VARIANT = FRAGMENT_PRICE + '\n        fragment ProductVariant on ProductVariant {\n          id\n          name\n          pricing {\n            onSale\n            priceUndiscounted {\n              ...Price\n            }\n            price {\n              ...Price\n            }\n          }\n          product {\n            id\n            name\n            thumbnail {\n              url\n              alt\n            }\n            thumbnail2x: thumbnail(size: 510) {\n              url\n            }\n          }\n        }\n    '
FRAGMENT_CHECKOUT_LINE = FRAGMENT_PRODUCT_VARIANT + '\n        fragment CheckoutLine on CheckoutLine {\n          id\n          quantity\n          totalPrice {\n            ...Price\n          }\n          variant {\n            ...ProductVariant\n          }\n          quantity\n        }\n    '
FRAGMENT_ADDRESS = '\n    fragment Address on Address {\n      id\n      firstName\n      lastName\n      companyName\n      streetAddress1\n      streetAddress2\n      city\n      postalCode\n      country {\n        code\n        country\n      }\n      countryArea\n      phone\n      isDefaultBillingAddress\n      isDefaultShippingAddress\n    }\n'
FRAGMENT_SHIPPING_METHOD = '\n    fragment ShippingMethod on ShippingMethod {\n        id\n        name\n        price {\n            amount\n        }\n    }\n'
FRAGMENT_COLLECTION_POINT = '\n   fragment CollectionPoint on Warehouse {\n        id\n        name\n        isPrivate\n        clickAndCollectOption\n        address {\n             streetAddress1\n          }\n     }\n'
FRAGMENT_CHECKOUT = FRAGMENT_CHECKOUT_LINE + FRAGMENT_ADDRESS + FRAGMENT_SHIPPING_METHOD + '\n        fragment Checkout on Checkout {\n          availablePaymentGateways {\n            id\n            name\n            config {\n              field\n              value\n            }\n          }\n          token\n          id\n          totalPrice {\n            ...Price\n          }\n          subtotalPrice {\n            ...Price\n          }\n          billingAddress {\n            ...Address\n          }\n          shippingAddress {\n            ...Address\n          }\n          email\n          availableShippingMethods {\n            ...ShippingMethod\n          }\n          shippingMethod {\n            ...ShippingMethod\n          }\n          shippingPrice {\n            ...Price\n          }\n          lines {\n            ...CheckoutLine\n          }\n          stockReservationExpires\n          isShippingRequired\n          discount {\n            currency\n            amount\n          }\n          discountName\n          translatedDiscountName\n          voucherCode\n          displayGrossPrices\n        }\n    '
FRAGMENT_CHECKOUT_FOR_CC = FRAGMENT_CHECKOUT_LINE + FRAGMENT_ADDRESS + FRAGMENT_SHIPPING_METHOD + FRAGMENT_COLLECTION_POINT + '\n        fragment Checkout on Checkout {\n          availablePaymentGateways {\n            id\n            name\n            config {\n              field\n              value\n            }\n          }\n          token\n          id\n          totalPrice {\n            ...Price\n          }\n          subtotalPrice {\n            ...Price\n          }\n          billingAddress {\n            ...Address\n          }\n          shippingAddress {\n            ...Address\n          }\n          email\n          availableShippingMethods {\n            ...ShippingMethod\n          }\n          availableCollectionPoints {\n            ...CollectionPoint\n          }\n          deliveryMethod {\n            __typename\n            ... on ShippingMethod {\n              ...ShippingMethod\n            }\n            ... on Warehouse {\n              ...CollectionPoint\n            }\n          }\n          shippingPrice {\n            ...Price\n          }\n          lines {\n            ...CheckoutLine\n          }\n          isShippingRequired\n          discount {\n            currency\n            amount\n          }\n          discountName\n          translatedDiscountName\n          voucherCode\n        }\n    '
MUTATION_CHECKOUT_CREATE = FRAGMENT_CHECKOUT + '\n        mutation CreateCheckout($checkoutInput: CheckoutCreateInput!) {\n            checkoutCreate(input: $checkoutInput) {\n                errors {\n                    field\n                    message\n                }\n                checkout {\n                    ...Checkout\n                }\n            }\n        }\n    '

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_create_checkout(api_client, graphql_address_data, stock, channel_USD, product_with_default_variant, product_with_single_variant, product_with_two_variants, count_queries):
    if False:
        for i in range(10):
            print('nop')
    checkout_counts = Checkout.objects.count()
    variables = {'checkoutInput': {'channel': channel_USD.slug, 'email': 'test@example.com', 'shippingAddress': graphql_address_data, 'lines': [{'quantity': 1, 'variantId': Node.to_global_id('ProductVariant', stock.product_variant.pk)}, {'quantity': 2, 'variantId': Node.to_global_id('ProductVariant', product_with_default_variant.variants.first().pk)}, {'quantity': 10, 'variantId': Node.to_global_id('ProductVariant', product_with_single_variant.variants.first().pk)}, {'quantity': 3, 'variantId': Node.to_global_id('ProductVariant', product_with_two_variants.variants.first().pk)}, {'quantity': 2, 'variantId': Node.to_global_id('ProductVariant', product_with_two_variants.variants.last().pk)}]}}
    get_graphql_content(api_client.post_graphql(MUTATION_CHECKOUT_CREATE, variables))
    assert checkout_counts + 1 == Checkout.objects.count()

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_create_checkout_with_reservations(site_settings_with_reservations, api_client, product, stock, warehouse, graphql_address_data, channel_USD, django_assert_num_queries, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = FRAGMENT_CHECKOUT_LINE + '\n            mutation CreateCheckout($checkoutInput: CheckoutCreateInput!) {\n              checkoutCreate(input: $checkoutInput) {\n                errors {\n                  field\n                  message\n                }\n                checkout {\n                  lines {\n                    ...CheckoutLine\n                  }\n                  stockReservationExpires\n                }\n              }\n            }\n        '
    variants = ProductVariant.objects.bulk_create([ProductVariant(product=product, sku=f'SKU_A_{i}') for i in range(10)])
    ProductVariantChannelListing.objects.bulk_create([ProductVariantChannelListing(variant=variant, channel=channel_USD, price_amount=Decimal(10), discounted_price_amount=Decimal(10), cost_price_amount=Decimal(1), currency=channel_USD.currency_code) for variant in variants])
    Stock.objects.bulk_create([Stock(product_variant=variant, warehouse=warehouse, quantity=15) for variant in variants])
    new_lines = []
    for variant in variants:
        variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
        new_lines.append({'quantity': 2, 'variantId': variant_id})
    test_email = 'test@example.com'
    shipping_address = graphql_address_data
    variables = {'checkoutInput': {'channel': channel_USD.slug, 'lines': [new_lines[0]], 'email': test_email, 'shippingAddress': shipping_address}}
    with django_assert_num_queries(61):
        response = api_client.post_graphql(query, variables)
        assert get_graphql_content(response)['data']['checkoutCreate']
        assert Checkout.objects.first().lines.count() == 1
    Checkout.objects.all().delete()
    test_email = 'test@example.com'
    shipping_address = graphql_address_data
    variables = {'checkoutInput': {'channel': channel_USD.slug, 'lines': new_lines, 'email': test_email, 'shippingAddress': shipping_address}}
    with django_assert_num_queries(61):
        response = api_client.post_graphql(query, variables)
        assert get_graphql_content(response)['data']['checkoutCreate']
        assert Checkout.objects.first().lines.count() == 10

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_add_shipping_to_checkout(api_client, checkout_with_shipping_address, shipping_method, count_queries):
    if False:
        while True:
            i = 10
    query = FRAGMENT_CHECKOUT + '\n            mutation updateCheckoutShippingOptions(\n              $id: ID, $shippingMethodId: ID\n            ) {\n              checkoutShippingMethodUpdate(\n                id: $id, shippingMethodId: $shippingMethodId\n              ) {\n                errors {\n                  field\n                  message\n                }\n                checkout {\n                  ...Checkout\n                }\n              }\n            }\n        '
    variables = {'id': to_global_id_or_none(checkout_with_shipping_address), 'shippingMethodId': Node.to_global_id('ShippingMethod', shipping_method.pk)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutShippingMethodUpdate']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_add_delivery_to_checkout(api_client, checkout_with_item_for_cc, warehouse_for_cc, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = FRAGMENT_CHECKOUT + '\n            mutation updateCheckoutDeliveryOptions(\n              $id: ID, $deliveryMethodId: ID\n            ) {\n              checkoutDeliveryMethodUpdate(\n                id: $id, deliveryMethodId: $deliveryMethodId\n              ) {\n                errors {\n                  field\n                  message\n                }\n                checkout {\n                  ...Checkout\n                }\n              }\n            }\n        '
    variables = {'id': to_global_id_or_none(checkout_with_item_for_cc), 'deliveryMethodId': Node.to_global_id('Warehouse', warehouse_for_cc.pk)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutDeliveryMethodUpdate']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_add_billing_address_to_checkout(api_client, graphql_address_data, checkout_with_shipping_method, count_queries):
    if False:
        print('Hello World!')
    query = FRAGMENT_CHECKOUT + '\n            mutation UpdateCheckoutBillingAddress(\n              $id: ID, $billingAddress: AddressInput!\n            ) {\n              checkoutBillingAddressUpdate(\n                  id: $id, billingAddress: $billingAddress\n              ) {\n                errors {\n                  field\n                  message\n                }\n                checkout {\n                  ...Checkout\n                }\n              }\n            }\n        '
    variables = {'id': to_global_id_or_none(checkout_with_shipping_method), 'billingAddress': graphql_address_data}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutBillingAddressUpdate']['errors']
MUTATION_CHECKOUT_LINES_UPDATE = FRAGMENT_CHECKOUT_LINE + '\n        mutation updateCheckoutLine($id: ID, $lines: [CheckoutLineUpdateInput!]!){\n          checkoutLinesUpdate(id: $id, lines: $lines) {\n            checkout {\n              id\n              lines {\n                ...CheckoutLine\n              }\n              totalPrice {\n                ...Price\n              }\n              subtotalPrice {\n                ...Price\n              }\n              isShippingRequired\n            }\n            errors {\n                field\n                message\n            }\n          }\n        }\n    '

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_update_checkout_lines(api_client, checkout_with_items, stock, product_with_default_variant, product_with_single_variant, product_with_two_variants, count_queries):
    if False:
        return 10
    variables = {'id': to_global_id_or_none(checkout_with_items), 'lines': [{'quantity': 1, 'variantId': Node.to_global_id('ProductVariant', stock.product_variant.pk)}, {'quantity': 2, 'variantId': Node.to_global_id('ProductVariant', product_with_default_variant.variants.first().pk)}, {'quantity': 10, 'variantId': Node.to_global_id('ProductVariant', product_with_single_variant.variants.first().pk)}, {'quantity': 3, 'variantId': Node.to_global_id('ProductVariant', product_with_two_variants.variants.first().pk)}, {'quantity': 2, 'variantId': Node.to_global_id('ProductVariant', product_with_two_variants.variants.last().pk)}]}
    response = get_graphql_content(api_client.post_graphql(MUTATION_CHECKOUT_LINES_UPDATE, variables))
    assert not response['data']['checkoutLinesUpdate']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_update_checkout_lines_with_reservations(site_settings_with_reservations, user_api_client, channel_USD, checkout_with_item, product, stock, warehouse, django_assert_num_queries, count_queries):
    if False:
        print('Hello World!')
    checkout = checkout_with_item
    variants = ProductVariant.objects.bulk_create([ProductVariant(product=product, sku=f'SKU_A_{i}') for i in range(10)])
    ProductVariantChannelListing.objects.bulk_create([ProductVariantChannelListing(variant=variant, channel=channel_USD, price_amount=Decimal(10), discounted_price_amount=Decimal(10), cost_price_amount=Decimal(1), currency=channel_USD.currency_code) for variant in variants])
    Stock.objects.bulk_create([Stock(product_variant=variant, warehouse=warehouse, quantity=15) for variant in variants])
    add_variants_to_checkout(checkout, variants, [CheckoutLineData(variant_id=str(variant.pk), quantity=2, quantity_to_update=True, custom_price=None, custom_price_to_update=False) for variant in variants], channel_USD, replace_reservations=True, reservation_length=5)
    with django_assert_num_queries(75):
        variant_id = graphene.Node.to_global_id('ProductVariant', variants[0].pk)
        variables = {'id': to_global_id_or_none(checkout), 'lines': [{'quantity': 3, 'variantId': variant_id}]}
        response = user_api_client.post_graphql(MUTATION_CHECKOUT_LINES_UPDATE, variables)
        content = get_graphql_content(response)
        data = content['data']['checkoutLinesUpdate']
        assert not data['errors']
    with django_assert_num_queries(75):
        variables = {'id': to_global_id_or_none(checkout), 'lines': []}
        for variant in variants:
            variant_id = graphene.Node.to_global_id('ProductVariant', variant.pk)
            variables['lines'].append({'quantity': 4, 'variantId': variant_id})
        response = user_api_client.post_graphql(MUTATION_CHECKOUT_LINES_UPDATE, variables)
        content = get_graphql_content(response)
        data = content['data']['checkoutLinesUpdate']
        assert not data['errors']
MUTATION_CHECKOUT_LINES_ADD = FRAGMENT_CHECKOUT_LINE + '\n        mutation addCheckoutLines($id: ID, $lines: [CheckoutLineInput!]!){\n          checkoutLinesAdd(id: $id, lines: $lines) {\n            checkout {\n              id\n              lines {\n                ...CheckoutLine\n              }\n            }\n            errors {\n              field\n              message\n            }\n          }\n        }\n    '

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
@patch('saleor.webhook.transport.synchronous.transport.send_webhook_request_sync')
def test_add_checkout_lines(mock_send_request, api_client, checkout_with_single_item, stock, product_with_default_variant, product_with_single_variant, product_with_two_variants, count_queries, shipping_app, settings):
    if False:
        return 10
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    mock_json_response = [{'id': 'abcd', 'name': 'Provider - Economy', 'amount': '10', 'currency': 'USD', 'maximum_delivery_days': '7'}]
    mock_send_request.return_value = mock_json_response
    variables = {'id': Node.to_global_id('Checkout', checkout_with_single_item.pk), 'lines': [{'quantity': 1, 'variantId': Node.to_global_id('ProductVariant', stock.product_variant.pk)}, {'quantity': 2, 'variantId': Node.to_global_id('ProductVariant', product_with_default_variant.variants.first().pk)}, {'quantity': 10, 'variantId': Node.to_global_id('ProductVariant', product_with_single_variant.variants.first().pk)}, {'quantity': 3, 'variantId': Node.to_global_id('ProductVariant', product_with_two_variants.variants.first().pk)}, {'quantity': 2, 'variantId': Node.to_global_id('ProductVariant', product_with_two_variants.variants.last().pk)}]}
    response = get_graphql_content(api_client.post_graphql(MUTATION_CHECKOUT_LINES_ADD, variables))
    assert not response['data']['checkoutLinesAdd']['errors']
    assert mock_send_request.call_count == 0

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
@patch('saleor.webhook.transport.synchronous.transport.send_webhook_request_sync')
def test_add_checkout_lines_with_external_shipping(mock_send_request, api_client, checkout_with_single_item, address, stock, product_with_default_variant, product_with_single_variant, product_with_two_variants, count_queries, shipping_app, settings):
    if False:
        i = 10
        return i + 15
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    response_method_id = 'abcd'
    mock_json_response = [{'id': response_method_id, 'name': 'Provider - Economy', 'amount': '10', 'currency': 'USD', 'maximum_delivery_days': '7'}]
    mock_send_request.return_value = mock_json_response
    external_shipping_method_id = Node.to_global_id('app', f'{shipping_app.id}:{response_method_id}')
    checkout_with_single_item.shipping_address = address
    set_external_shipping_id(checkout_with_single_item, external_shipping_method_id)
    checkout_with_single_item.save()
    checkout_with_single_item.metadata_storage.save()
    variables = {'id': Node.to_global_id('Checkout', checkout_with_single_item.pk), 'lines': [{'quantity': 1, 'variantId': Node.to_global_id('ProductVariant', stock.product_variant.pk)}, {'quantity': 2, 'variantId': Node.to_global_id('ProductVariant', product_with_default_variant.variants.first().pk)}, {'quantity': 10, 'variantId': Node.to_global_id('ProductVariant', product_with_single_variant.variants.first().pk)}, {'quantity': 3, 'variantId': Node.to_global_id('ProductVariant', product_with_two_variants.variants.first().pk)}, {'quantity': 2, 'variantId': Node.to_global_id('ProductVariant', product_with_two_variants.variants.last().pk)}]}
    response = get_graphql_content(api_client.post_graphql(MUTATION_CHECKOUT_LINES_ADD, variables))
    assert not response['data']['checkoutLinesAdd']['errors']
    assert mock_send_request.call_count == 2

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_add_checkout_lines_with_reservations(site_settings_with_reservations, user_api_client, channel_USD, checkout_with_item, product, stock, warehouse, django_assert_num_queries, count_queries):
    if False:
        return 10
    checkout = checkout_with_item
    line = checkout.lines.first()
    variants = ProductVariant.objects.bulk_create([ProductVariant(product=product, sku=f'SKU_A_{i}') for i in range(10)])
    ProductVariantChannelListing.objects.bulk_create([ProductVariantChannelListing(variant=variant, channel=channel_USD, price_amount=Decimal(10), discounted_price_amount=Decimal(10), cost_price_amount=Decimal(1), currency=channel_USD.currency_code) for variant in variants])
    Stock.objects.bulk_create([Stock(product_variant=variant, warehouse=warehouse, quantity=15) for variant in variants])
    new_lines = []
    for variant in variants:
        variant_id = graphene.Node.to_global_id('ProductVariant', variant.id)
        new_lines.append({'quantity': 2, 'variantId': variant_id})
    with django_assert_num_queries(74):
        variables = {'id': Node.to_global_id('Checkout', checkout.pk), 'lines': [new_lines[0]], 'channelSlug': checkout.channel.slug}
        response = user_api_client.post_graphql(MUTATION_CHECKOUT_LINES_ADD, variables)
        content = get_graphql_content(response)
        data = content['data']['checkoutLinesAdd']
        assert not data['errors']
    checkout.lines.exclude(id=line.id).delete()
    with django_assert_num_queries(74):
        variables = {'id': Node.to_global_id('Checkout', checkout.pk), 'lines': new_lines, 'channelSlug': checkout.channel.slug}
        response = user_api_client.post_graphql(MUTATION_CHECKOUT_LINES_ADD, variables)
        content = get_graphql_content(response)
        data = content['data']['checkoutLinesAdd']
        assert not data['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_checkout_shipping_address_update(api_client, graphql_address_data, checkout_with_variants, count_queries):
    if False:
        print('Hello World!')
    query = FRAGMENT_CHECKOUT + '\n            mutation UpdateCheckoutShippingAddress(\n              $id: ID, $shippingAddress: AddressInput!\n            ) {\n              checkoutShippingAddressUpdate(\n                id: $id, shippingAddress: $shippingAddress\n              ) {\n                errors {\n                  field\n                  message\n                }\n                checkout {\n                  ...Checkout\n                }\n              }\n            }\n        '
    variables = {'id': to_global_id_or_none(checkout_with_variants), 'shippingAddress': graphql_address_data}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutShippingAddressUpdate']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_checkout_email_update(api_client, checkout_with_variants, count_queries):
    if False:
        return 10
    query = FRAGMENT_CHECKOUT + '\n            mutation UpdateCheckoutEmail(\n              $id: ID, $email: String!\n            ) {\n              checkoutEmailUpdate(id: $id, email: $email) {\n                checkout {\n                  ...Checkout\n                }\n                errors {\n                  field\n                  message\n                }\n              }\n            }\n        '
    variables = {'id': to_global_id_or_none(checkout_with_variants), 'email': 'newEmail@example.com'}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutEmailUpdate']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_checkout_voucher_code(api_client, checkout_with_billing_address, voucher, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = FRAGMENT_CHECKOUT + '\n            mutation AddCheckoutPromoCode($id: ID, $promoCode: String!) {\n              checkoutAddPromoCode(id: $id, promoCode: $promoCode) {\n                checkout {\n                  ...Checkout\n                }\n                errors {\n                  field\n                  message\n                }\n                errors {\n                  field\n                  message\n                  code\n                }\n              }\n            }\n        '
    variables = {'id': to_global_id_or_none(checkout_with_billing_address), 'promoCode': voucher.code}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutAddPromoCode']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_checkout_payment_charge(api_client, checkout_with_billing_address, count_queries):
    if False:
        while True:
            i = 10
    query = '\n        mutation createPayment($input: PaymentInput!, $id: ID) {\n          checkoutPaymentCreate(input: $input, id: $id) {\n            errors {\n              field\n              message\n            }\n          }\n        }\n    '
    manager = get_plugins_manager()
    (lines, _) = fetch_checkout_lines(checkout_with_billing_address)
    checkout_info = fetch_checkout_info(checkout_with_billing_address, lines, manager)
    manager = get_plugins_manager()
    total = calculations.checkout_total(manager=manager, checkout_info=checkout_info, lines=lines, address=checkout_with_billing_address.shipping_address)
    variables = {'id': to_global_id_or_none(checkout_with_billing_address), 'input': {'amount': total.gross.amount, 'gateway': 'mirumee.payments.dummy', 'token': 'charged'}}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutPaymentCreate']['errors']
ORDER_PRICE_FRAGMENT = '\nfragment OrderPrice on TaxedMoney {\n  gross {\n    amount\n    currency\n    __typename\n  }\n  net {\n    amount\n    currency\n    __typename\n  }\n  __typename\n}\n'
FRAGMENT_ORDER_DETAIL = FRAGMENT_ADDRESS + FRAGMENT_PRODUCT_VARIANT + ORDER_PRICE_FRAGMENT + '\n  fragment OrderDetail on Order {\n    userEmail\n    paymentStatus\n    paymentStatusDisplay\n    status\n    statusDisplay\n    id\n    token\n    number\n    shippingAddress {\n      ...Address\n      __typename\n    }\n    lines {\n      productName\n      quantity\n      variant {\n        ...ProductVariant\n        __typename\n      }\n      unitPrice {\n        currency\n        ...OrderPrice\n        __typename\n      }\n      totalPrice {\n        currency\n        ...OrderPrice\n        __typename\n      }\n      __typename\n    }\n    subtotal {\n      ...OrderPrice\n      __typename\n    }\n    total {\n      ...OrderPrice\n      __typename\n    }\n    shippingPrice {\n      ...OrderPrice\n      __typename\n    }\n    __typename\n  }\n  '
FRAGMENT_ORDER_DETAIL_FOR_CC = FRAGMENT_ADDRESS + FRAGMENT_PRODUCT_VARIANT + ORDER_PRICE_FRAGMENT + FRAGMENT_COLLECTION_POINT + FRAGMENT_SHIPPING_METHOD + '\n  fragment OrderDetail on Order {\n    userEmail\n    paymentStatus\n    paymentStatusDisplay\n    status\n    statusDisplay\n    id\n    token\n    number\n    shippingAddress {\n      ...Address\n      __typename\n    }\n    deliveryMethod {\n      __typename\n      ... on ShippingMethod {\n        ...ShippingMethod\n      }\n      ... on Warehouse {\n        ...CollectionPoint\n      }\n    }\n    lines {\n      productName\n      quantity\n      variant {\n        ...ProductVariant\n        __typename\n      }\n      unitPrice {\n        currency\n        ...OrderPrice\n        __typename\n      }\n      totalPrice {\n        currency\n        ...OrderPrice\n        __typename\n      }\n      __typename\n    }\n    subtotal {\n      ...OrderPrice\n      __typename\n    }\n    total {\n      ...OrderPrice\n      __typename\n    }\n    shippingPrice {\n      ...OrderPrice\n      __typename\n    }\n    __typename\n  }\n  '
COMPLETE_CHECKOUT_MUTATION_PART = '\n    mutation completeCheckout($id: ID) {\n      checkoutComplete(id: $id) {\n        errors {\n          code\n          field\n          message\n        }\n        order {\n          ...OrderDetail\n          __typename\n        }\n        confirmationNeeded\n        confirmationData\n      }\n    }\n'
COMPLETE_CHECKOUT_MUTATION = FRAGMENT_ORDER_DETAIL + COMPLETE_CHECKOUT_MUTATION_PART
COMPLETE_CHECKOUT_MUTATION_FOR_CC = FRAGMENT_ORDER_DETAIL_FOR_CC + COMPLETE_CHECKOUT_MUTATION_PART

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_complete_checkout(api_client, checkout_with_charged_payment, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = COMPLETE_CHECKOUT_MUTATION
    variables = {'id': to_global_id_or_none(checkout_with_charged_payment)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutComplete']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
@patch('saleor.plugins.manager.PluginsManager.product_variant_out_of_stock')
def test_complete_checkout_with_out_of_stock_webhook(product_variant_out_of_stock_webhook_mock, api_client, checkout_with_charged_payment, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = COMPLETE_CHECKOUT_MUTATION
    Stock.objects.update(quantity=10)
    variables = {'id': to_global_id_or_none(checkout_with_charged_payment)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutComplete']['errors']
    product_variant_out_of_stock_webhook_mock.assert_called_once_with(Stock.objects.last())

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_complete_checkout_with_single_line(api_client, checkout_with_charged_payment, count_queries):
    if False:
        print('Hello World!')
    query = COMPLETE_CHECKOUT_MUTATION
    checkout_with_charged_payment.lines.set([checkout_with_charged_payment.lines.first()])
    variables = {'id': to_global_id_or_none(checkout_with_charged_payment)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutComplete']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_complete_checkout_with_digital_line(api_client, checkout_with_digital_line_with_charged_payment, count_queries):
    if False:
        print('Hello World!')
    query = COMPLETE_CHECKOUT_MUTATION
    variables = {'id': to_global_id_or_none(checkout_with_digital_line_with_charged_payment)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutComplete']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_customer_complete_checkout(api_client, checkout_with_charged_payment, customer_user, count_queries):
    if False:
        while True:
            i = 10
    query = COMPLETE_CHECKOUT_MUTATION
    checkout = checkout_with_charged_payment
    checkout.user = customer_user
    checkout.save()
    variables = {'id': to_global_id_or_none(checkout)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutComplete']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_customer_complete_checkout_for_cc(api_client, checkout_with_charged_payment_for_cc, customer_user, count_queries):
    if False:
        while True:
            i = 10
    query = COMPLETE_CHECKOUT_MUTATION_FOR_CC
    checkout = checkout_with_charged_payment_for_cc
    checkout.user = customer_user
    checkout.save()
    variables = {'id': to_global_id_or_none(checkout)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutComplete']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_complete_checkout_preorder(api_client, checkout_preorder_with_charged_payment, count_queries):
    if False:
        while True:
            i = 10
    query = COMPLETE_CHECKOUT_MUTATION
    variables = {'id': to_global_id_or_none(checkout_preorder_with_charged_payment)}
    response = get_graphql_content(api_client.post_graphql(query, variables))
    assert not response['data']['checkoutComplete']['errors']
MUTATION_CHECKOUT_CREATE_FROM_ORDER = FRAGMENT_CHECKOUT + '\nmutation CheckoutCreateFromOrder($id: ID!) {\n  checkoutCreateFromOrder(id:$id){\n    errors{\n      field\n      message\n      code\n    }\n    unavailableVariants{\n      message\n      code\n      variantId\n      lineId\n    }\n    checkout{\n      ...Checkout\n    }\n  }\n}\n'

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_checkout_create_from_order(user_api_client, order_with_lines):
    if False:
        while True:
            i = 10
    order_with_lines.user = user_api_client.user
    order_with_lines.save()
    Stock.objects.update(quantity=10)
    variables = {'id': graphene.Node.to_global_id('Order', order_with_lines.pk)}
    response = user_api_client.post_graphql(MUTATION_CHECKOUT_CREATE_FROM_ORDER, variables)
    content = get_graphql_content(response)
    assert not content['data']['checkoutCreateFromOrder']['errors']

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_checkout_gift_cards(staff_api_client, checkout_with_gift_card, checkout_with_gift_card_items, gift_card_created_by_staff, gift_card, permission_manage_gift_card, permission_manage_checkouts):
    if False:
        for i in range(10):
            print('nop')
    checkout_with_gift_card.gift_cards.add(gift_card_created_by_staff)
    checkout_with_gift_card.gift_cards.add(gift_card)
    checkout_with_gift_card.save()
    checkout_with_gift_card_items.gift_cards.add(gift_card_created_by_staff)
    checkout_with_gift_card_items.gift_cards.add(gift_card)
    checkout_with_gift_card_items.save()
    response = staff_api_client.post_graphql(CHECKOUT_GIFT_CARD_QUERY, {}, permissions=[permission_manage_gift_card, permission_manage_checkouts], check_no_permissions=False)
    assert response.status_code == 200