from decimal import Decimal
import graphene
import pytest
from ....tax import TaxCalculationStrategy
from ....tax.models import TaxClassCountryRate, TaxConfigurationPerCountry
from ...tests.utils import get_graphql_content
TAX_RATE_DE = 19
TAX_RATE_PL = 23
FRAGMENT_PRICE = '\n  fragment Price on TaxedMoney {\n    gross {\n      amount\n    }\n    net {\n      amount\n    }\n    tax {\n      amount\n    }\n  }\n'
FRAGMENT_PRICING = '\n  fragment Pricing on ProductPricingInfo {\n    priceRange {\n      start {\n        ...Price\n      }\n      stop {\n        ...Price\n      }\n    }\n    priceRangeUndiscounted {\n      start {\n        ...Price\n      }\n      stop {\n        ...Price\n      }\n    }\n  }\n' + FRAGMENT_PRICE
QUERY_PRODUCT_PRICING = '\n  query Product($id: ID!, $channel: String!) {\n    product(id: $id, channel: $channel) {\n      pricingPL: pricing(address: { country: PL }) {\n        ...Pricing\n      }\n      pricingDE: pricing(address: { country: DE }) {\n        ...Pricing\n      }\n      pricing: pricing {\n        ...Pricing\n      }\n    }\n  }\n' + FRAGMENT_PRICING

def _enable_flat_rates(channel, prices_entered_with_tax):
    if False:
        while True:
            i = 10
    tc = channel.tax_configuration
    tc.tax_calculation_strategy = TaxCalculationStrategy.FLAT_RATES
    tc.prices_entered_with_tax = prices_entered_with_tax
    tc.charge_taxes = True
    tc.country_exceptions.all().delete()
    tc.country_exceptions.create(country='PL', charge_taxes=True, tax_calculation_strategy=TaxCalculationStrategy.FLAT_RATES)
    tc.country_exceptions.create(country='DE', charge_taxes=True, tax_calculation_strategy=TaxCalculationStrategy.FLAT_RATES)
    tc.save()

def _configure_tax_rates(product):
    if False:
        while True:
            i = 10
    product.tax_class.country_rates.all().delete()
    product.tax_class.country_rates.create(country='PL', rate=TAX_RATE_PL)
    product.tax_class.country_rates.create(country='DE', rate=TAX_RATE_DE)

@pytest.mark.parametrize(('net_PL', 'gross_PL', 'net_DE', 'gross_DE', 'prices_entered_with_tax'), [(40.65, 50.0, 42.02, 50.0, True), (50.0, 61.5, 50.0, 59.5, False)])
def test_product_pricing(product_available_in_many_channels, channel_PLN, user_api_client, net_PL, gross_PL, net_DE, gross_DE, prices_entered_with_tax):
    if False:
        return 10
    product = product_available_in_many_channels
    _enable_flat_rates(channel_PLN, prices_entered_with_tax)
    _configure_tax_rates(product)
    variables = {'id': graphene.Node.to_global_id('Product', product.id), 'channel': channel_PLN.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_PRICING, variables)
    content = get_graphql_content(response)
    data = content['data']['product']
    price_range_PL = data['pricingPL']['priceRange']
    price_range_undiscounted_PL = data['pricingPL']['priceRangeUndiscounted']
    assert price_range_PL['start']['net']['amount'] == net_PL
    assert price_range_PL['start']['gross']['amount'] == gross_PL
    assert price_range_PL['stop']['net']['amount'] == net_PL
    assert price_range_PL['stop']['gross']['amount'] == gross_PL
    assert price_range_undiscounted_PL['start']['net']['amount'] == net_PL
    assert price_range_undiscounted_PL['start']['gross']['amount'] == gross_PL
    assert price_range_undiscounted_PL['stop']['net']['amount'] == net_PL
    assert price_range_undiscounted_PL['stop']['gross']['amount'] == gross_PL
    price_range_DE = data['pricingDE']['priceRange']
    price_range_undiscounted_DE = data['pricingDE']['priceRangeUndiscounted']
    assert price_range_DE['start']['net']['amount'] == net_DE
    assert price_range_DE['start']['gross']['amount'] == gross_DE
    assert price_range_DE['stop']['net']['amount'] == net_DE
    assert price_range_DE['stop']['gross']['amount'] == gross_DE
    assert price_range_undiscounted_DE['start']['net']['amount'] == net_DE
    assert price_range_undiscounted_DE['start']['gross']['amount'] == gross_DE
    assert price_range_undiscounted_DE['stop']['net']['amount'] == net_DE
    assert price_range_undiscounted_DE['stop']['gross']['amount'] == gross_DE

def test_product_pricing_default_country_default_rate(product_available_in_many_channels, channel_PLN, user_api_client):
    if False:
        print('Hello World!')
    product = product_available_in_many_channels
    _enable_flat_rates(channel_PLN, True)
    TaxClassCountryRate.objects.all().delete()
    TaxClassCountryRate.objects.create(country=channel_PLN.default_country, rate=TAX_RATE_PL)
    variables = {'id': graphene.Node.to_global_id('Product', product.id), 'channel': channel_PLN.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_PRICING, variables)
    content = get_graphql_content(response)
    data = content['data']['product']
    variant = product.variants.first()
    channel_listing = variant.channel_listings.filter(channel_id=channel_PLN.id).first()
    gross = channel_listing.price_amount.quantize(Decimal('.01'))
    net = (gross / Decimal(1 + TAX_RATE_PL / 100)).quantize(Decimal('.01'))
    gross = float(gross)
    net = float(net)
    price_range = data['pricing']['priceRange']
    price_range_undiscounted = data['pricing']['priceRangeUndiscounted']
    assert price_range['start']['net']['amount'] == net
    assert price_range['start']['gross']['amount'] == gross
    assert price_range['stop']['net']['amount'] == net
    assert price_range['stop']['gross']['amount'] == gross
    assert price_range_undiscounted['start']['net']['amount'] == net
    assert price_range_undiscounted['start']['gross']['amount'] == gross
    assert price_range_undiscounted['stop']['net']['amount'] == net
    assert price_range_undiscounted['stop']['gross']['amount'] == gross

def test_product_pricing_use_tax_class_from_product_type(product_available_in_many_channels, channel_PLN, user_api_client):
    if False:
        print('Hello World!')
    product = product_available_in_many_channels
    _enable_flat_rates(channel_PLN, True)
    TaxClassCountryRate.objects.all().delete()
    product.tax_class = None
    product.save(update_fields=['tax_class'])
    product.product_type.tax_class.country_rates.create(country=channel_PLN.default_country, rate=TAX_RATE_PL)
    variables = {'id': graphene.Node.to_global_id('Product', product.id), 'channel': channel_PLN.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_PRICING, variables)
    content = get_graphql_content(response)
    data = content['data']['product']
    variant = product.variants.first()
    channel_listing = variant.channel_listings.filter(channel_id=channel_PLN.id).first()
    gross = channel_listing.price_amount.quantize(Decimal('.01'))
    net = (gross / Decimal(1 + TAX_RATE_PL / 100)).quantize(Decimal('.01'))
    gross = float(gross)
    net = float(net)
    price_range_PL = data['pricingPL']['priceRange']
    price_range_undiscounted_PL = data['pricingPL']['priceRangeUndiscounted']
    assert price_range_PL['start']['net']['amount'] == net
    assert price_range_PL['start']['gross']['amount'] == gross
    assert price_range_PL['stop']['net']['amount'] == net
    assert price_range_PL['stop']['gross']['amount'] == gross
    assert price_range_undiscounted_PL['start']['net']['amount'] == net
    assert price_range_undiscounted_PL['start']['gross']['amount'] == gross
    assert price_range_undiscounted_PL['stop']['net']['amount'] == net
    assert price_range_undiscounted_PL['stop']['gross']['amount'] == gross

def test_product_pricing_no_flat_rates_in_one_country(product_available_in_many_channels, channel_PLN, user_api_client):
    if False:
        while True:
            i = 10
    product = product_available_in_many_channels
    _enable_flat_rates(channel_PLN, True)
    _configure_tax_rates(product)
    TaxConfigurationPerCountry.objects.filter(country='PL').update(tax_calculation_strategy='TAX_APP')
    variables = {'id': graphene.Node.to_global_id('Product', product.id), 'channel': channel_PLN.slug}
    response = user_api_client.post_graphql(QUERY_PRODUCT_PRICING, variables)
    content = get_graphql_content(response)
    data = content['data']['product']
    variant = product.variants.first()
    channel_listing = variant.channel_listings.filter(channel_id=channel_PLN.id).first()
    price_pl = float(channel_listing.price_amount.quantize(Decimal('.01')))
    gross_de = channel_listing.price_amount.quantize(Decimal('.01'))
    net_de = (gross_de / Decimal(1 + TAX_RATE_DE / 100)).quantize(Decimal('.01'))
    gross_de = float(gross_de)
    net_de = float(net_de)
    price_range_PL = data['pricingPL']['priceRange']
    price_range_undiscounted_PL = data['pricingPL']['priceRangeUndiscounted']
    assert price_range_PL['start']['net']['amount'] == price_pl
    assert price_range_PL['start']['gross']['amount'] == price_pl
    assert price_range_PL['stop']['net']['amount'] == price_pl
    assert price_range_PL['stop']['gross']['amount'] == price_pl
    assert price_range_undiscounted_PL['start']['net']['amount'] == price_pl
    assert price_range_undiscounted_PL['start']['gross']['amount'] == price_pl
    assert price_range_undiscounted_PL['stop']['net']['amount'] == price_pl
    assert price_range_undiscounted_PL['stop']['gross']['amount'] == price_pl
    price_range_DE = data['pricingDE']['priceRange']
    price_range_undiscounted_DE = data['pricingDE']['priceRangeUndiscounted']
    assert price_range_DE['start']['net']['amount'] == net_de
    assert price_range_DE['start']['gross']['amount'] == gross_de
    assert price_range_DE['stop']['net']['amount'] == net_de
    assert price_range_DE['stop']['gross']['amount'] == gross_de
    assert price_range_undiscounted_DE['start']['net']['amount'] == net_de
    assert price_range_undiscounted_DE['start']['gross']['amount'] == gross_de
    assert price_range_undiscounted_DE['stop']['net']['amount'] == net_de
    assert price_range_undiscounted_DE['stop']['gross']['amount'] == gross_de