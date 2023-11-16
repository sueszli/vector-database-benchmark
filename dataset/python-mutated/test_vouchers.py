import pytest
from .....discount.models import Voucher, VoucherChannelListing, VoucherCode
from ....tests.utils import get_graphql_content

@pytest.fixture
def vouchers_list(channel_USD, channel_PLN):
    if False:
        print('Hello World!')
    vouchers = Voucher.objects.bulk_create([Voucher(name='Voucher1'), Voucher(name='Voucher2'), Voucher(name='Voucher3')])
    VoucherCode.objects.bulk_create([VoucherCode(code='Voucher1', voucher=vouchers[0]), VoucherCode(code='Voucher2', voucher=vouchers[1]), VoucherCode(code='Voucher3', voucher=vouchers[2])])
    values = [15, 5, 25]
    voucher_channel_listings = []
    for (voucher, value) in zip(vouchers, values):
        voucher_channel_listings.append(VoucherChannelListing(voucher=voucher, channel=channel_USD, discount_value=value, currency=channel_USD.currency_code))
        voucher_channel_listings.append(VoucherChannelListing(voucher=voucher, channel=channel_PLN, discount_value=value * 2, currency=channel_PLN.currency_code))
    VoucherChannelListing.objects.bulk_create(voucher_channel_listings)
    return vouchers
VOUCHERS_QUERY = '\nquery GetVouchers($channel: String){\n  vouchers(last: 10, channel: $channel) {\n    edges {\n      node {\n        id\n        name\n        type\n        startDate\n        endDate\n        usageLimit\n        code\n        applyOncePerOrder\n        applyOncePerCustomer\n        discountValueType\n        minCheckoutItemsQuantity\n        countries{\n          code\n          country\n          vat{\n            countryCode\n            standardRate\n          }\n        }\n        categories(first: 10) {\n          edges {\n            node {\n              id\n            }\n          }\n        }\n        collections(first: 10) {\n          edges {\n            node {\n              id\n            }\n          }\n        }\n        products(first: 10) {\n          edges {\n            node {\n              id\n            }\n          }\n        }\n        variants(first: 10) {\n          edges {\n            node {\n              id\n            }\n          }\n        }\n        channelListings {\n          id\n          discountValue\n          currency\n          minSpent{\n            currency\n            amount\n          }\n          channel {\n            id\n            name\n            isActive\n            slug\n            currencyCode\n          }\n        }\n        discountValue\n        currency\n      }\n    }\n  }\n}\n'

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_vouchers_query_with_channel_slug(staff_api_client, vouchers_list, channel_USD, permission_manage_discounts, count_queries):
    if False:
        i = 10
        return i + 15
    variables = {'channel': channel_USD.slug}
    get_graphql_content(staff_api_client.post_graphql(VOUCHERS_QUERY, variables, permissions=[permission_manage_discounts], check_no_permissions=False))

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_vouchers_query_withot_channel_slug(staff_api_client, vouchers_list, permission_manage_discounts, count_queries):
    if False:
        i = 10
        return i + 15
    get_graphql_content(staff_api_client.post_graphql(VOUCHERS_QUERY, {}, permissions=[permission_manage_discounts], check_no_permissions=False))