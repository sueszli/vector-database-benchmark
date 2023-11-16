from unittest.mock import patch
import graphene
from .....discount import RewardValueType
from .....discount.error_codes import DiscountErrorCode
from ....tests.utils import assert_negative_positive_decimal_value, get_graphql_content
SALE_CHANNEL_LISTING_UPDATE_MUTATION = '\nmutation UpdateSaleChannelListing(\n    $id: ID!\n    $input: SaleChannelListingInput!\n) {\n    saleChannelListingUpdate(id: $id, input: $input) {\n        errors {\n            field\n            message\n            code\n            channels\n        }\n        sale {\n            name\n            channelListings {\n                discountValue\n                channel {\n                    slug\n                }\n            }\n        }\n    }\n}\n'

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_add_channels(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_PLN):
    if False:
        return 10
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    discounted = 1.12
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_id, 'discountValue': discounted}]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    assert not content['data']['saleChannelListingUpdate']['errors']
    data = content['data']['saleChannelListingUpdate']['sale']
    assert data['name'] == promotion.name
    channel_listing = data['channelListings']
    discounts = [item['discountValue'] for item in channel_listing]
    slugs = [item['channel']['slug'] for item in channel_listing]
    assert discounted in discounts
    assert channel_PLN.slug in slugs
    promotion.refresh_from_db()
    rules = promotion.rules.all()
    assert len(rules) == 2
    assert len({(rule.reward_value_type, str(rule.catalogue_predicate)) for rule in rules}) == 1
    assert all([rule.old_channel_listing_id for rule in rules])
    mock_update_products_discounted_prices_of_promotion_task.delay.assert_called_once_with(promotion.pk)

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_add_multiple_channels(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_PLN, channel_JPY):
    if False:
        for i in range(10):
            print('nop')
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_pln_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    channel_jpy_id = graphene.Node.to_global_id('Channel', channel_JPY.id)
    discounted = 5
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_pln_id, 'discountValue': discounted}, {'channelId': channel_jpy_id, 'discountValue': discounted}]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    assert not content['data']['saleChannelListingUpdate']['errors']
    data = content['data']['saleChannelListingUpdate']['sale']
    assert data['name'] == promotion.name
    channel_listing = data['channelListings']
    discounts = [item['discountValue'] for item in channel_listing]
    slugs = [item['channel']['slug'] for item in channel_listing]
    assert discounted in discounts
    assert channel_PLN.slug in slugs
    assert channel_JPY.slug in slugs
    promotion.refresh_from_db()
    rules = promotion.rules.all()
    assert len(rules) == 3
    old_channel_listing_ids = [rule.old_channel_listing_id for rule in rules]
    assert all(old_channel_listing_ids)
    assert len(set(old_channel_listing_ids)) == 3
    mock_update_products_discounted_prices_of_promotion_task.delay.assert_called_once_with(promotion.pk)

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_update_channels(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_USD):
    if False:
        i = 10
        return i + 15
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    discounted = 10.11
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_id, 'discountValue': discounted}]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    assert not content['data']['saleChannelListingUpdate']['errors']
    data = content['data']['saleChannelListingUpdate']['sale']
    channel_listing = data['channelListings']
    assert len(channel_listing) == 1
    assert channel_listing[0]['discountValue'] == discounted
    assert channel_listing[0]['channel']['slug'] == channel_USD.slug
    promotion.refresh_from_db()
    rules = promotion.rules.all()
    assert len(rules) == 1
    mock_update_products_discounted_prices_of_promotion_task.delay.assert_called_once_with(promotion.pk)

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_remove_channels(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale_with_many_channels, permission_manage_discounts, channel_USD, channel_PLN):
    if False:
        return 10
    promotion = promotion_converted_from_sale_with_many_channels
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': sale_id, 'input': {'removeChannels': [channel_id]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    assert not content['data']['saleChannelListingUpdate']['errors']
    data = content['data']['saleChannelListingUpdate']['sale']
    assert data['name'] == promotion.name
    channel_listing = data['channelListings']
    assert len(channel_listing) == 1
    assert channel_listing[0]['channel']['slug'] == channel_PLN.slug
    promotion.refresh_from_db()
    rules = promotion.rules.all()
    assert len(rules) == 1
    mock_update_products_discounted_prices_of_promotion_task.delay.assert_called_once_with(promotion.pk)

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_remove_all_channels(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale_with_many_channels, permission_manage_discounts, channel_USD, channel_PLN):
    if False:
        print('Hello World!')
    promotion = promotion_converted_from_sale_with_many_channels
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_ids = [graphene.Node.to_global_id('Channel', channel.id) for channel in [channel_USD, channel_PLN]]
    rule = promotion.rules.first()
    reward_value_type = rule.reward_value_type
    predicate = rule.catalogue_predicate
    variables = {'id': sale_id, 'input': {'removeChannels': channel_ids}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    assert not content['data']['saleChannelListingUpdate']['errors']
    data = content['data']['saleChannelListingUpdate']['sale']
    assert data['name'] == promotion.name
    assert not data['channelListings']
    promotion.refresh_from_db()
    rules = promotion.rules.all()
    assert len(rules) == 1
    assert rules[0].reward_value_type == reward_value_type
    assert rules[0].catalogue_predicate == predicate
    mock_update_products_discounted_prices_of_promotion_task.delay.assert_called_once_with(promotion.pk)

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_add_update_remove_channels(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale_with_many_channels, permission_manage_discounts, channel_PLN, channel_USD, channel_JPY):
    if False:
        return 10
    promotion = promotion_converted_from_sale_with_many_channels
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_usd_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_pln_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    channel_jpy_id = graphene.Node.to_global_id('Channel', channel_JPY.id)
    discounted = 5
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_usd_id, 'discountValue': discounted}, {'channelId': channel_jpy_id, 'discountValue': discounted}], 'removeChannels': [channel_pln_id]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    assert not content['data']['saleChannelListingUpdate']['errors']
    data = content['data']['saleChannelListingUpdate']['sale']
    assert data['name'] == promotion.name
    channel_listings = data['channelListings']
    assert len(channel_listings) == 2
    assert all([listing['channel']['slug'] in [channel_USD.slug, channel_JPY.slug] for listing in channel_listings])
    assert all([listing['discountValue'] == discounted for listing in channel_listings])
    promotion.refresh_from_db()
    rules = promotion.rules.all()
    assert len(rules) == 2
    for rule in rules:
        assert len(rule.channels.all()) == 1
    mock_update_products_discounted_prices_of_promotion_task.delay.assert_called_once_with(promotion.pk)

def test_sale_channel_listing_update_with_negative_discounted_value(staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_USD):
    if False:
        i = 10
        return i + 15
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    discounted_value = -10
    staff_api_client.user.user_permissions.add(permission_manage_discounts)
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_id, 'discountValue': discounted_value}]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables)
    assert_negative_positive_decimal_value(response)

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_update_duplicated_ids_in_add_and_remove(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_USD):
    if False:
        while True:
            i = 10
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    discounted = 10.11
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_id, 'discountValue': discounted}], 'removeChannels': [channel_id]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    errors = content['data']['saleChannelListingUpdate']['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'input'
    assert errors[0]['code'] == DiscountErrorCode.DUPLICATED_INPUT_ITEM.name
    assert errors[0]['channels'] == [channel_id]
    mock_update_products_discounted_prices_of_promotion_task.assert_not_called()

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_update_duplicated_channel_in_add(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_USD):
    if False:
        return 10
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    discounted = 10.11
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_id, 'discountValue': discounted}, {'channelId': channel_id, 'discountValue': discounted}]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    errors = content['data']['saleChannelListingUpdate']['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'addChannels'
    assert errors[0]['code'] == DiscountErrorCode.DUPLICATED_INPUT_ITEM.name
    assert errors[0]['channels'] == [channel_id]
    mock_update_products_discounted_prices_of_promotion_task.assert_not_called()

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_update_duplicated_channel_in_remove(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_USD):
    if False:
        print('Hello World!')
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': sale_id, 'input': {'removeChannels': [channel_id, channel_id]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    errors = content['data']['saleChannelListingUpdate']['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'removeChannels'
    assert errors[0]['code'] == DiscountErrorCode.DUPLICATED_INPUT_ITEM.name
    assert errors[0]['channels'] == [channel_id]
    mock_update_products_discounted_prices_of_promotion_task.assert_not_called()

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_update_with_invalid_decimal_places(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_USD):
    if False:
        while True:
            i = 10
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    discounted = 1.123
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_id, 'discountValue': discounted}]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    errors = content['data']['saleChannelListingUpdate']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == DiscountErrorCode.INVALID.name
    assert errors[0]['field'] == 'input'
    assert errors[0]['channels'] == [channel_id]
    mock_update_products_discounted_prices_of_promotion_task.assert_not_called()

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_update_with_invalid_percentage_value(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_USD):
    if False:
        i = 10
        return i + 15
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    rule = promotion.rules.first()
    rule.reward_value_type = RewardValueType.PERCENTAGE
    rule.save(update_fields=['reward_value_type'])
    discounted = 101
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_id, 'discountValue': discounted}]}}
    response = staff_api_client.post_graphql(SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    errors = content['data']['saleChannelListingUpdate']['errors']
    assert len(errors) == 1
    assert errors[0]['code'] == DiscountErrorCode.INVALID.name
    assert errors[0]['field'] == 'input'
    assert errors[0]['channels'] == [channel_id]
    mock_update_products_discounted_prices_of_promotion_task.assert_not_called()
SALE_AND_SALE_CHANNEL_LISTING_UPDATE_MUTATION = '\nmutation UpdateSaleChannelListing(\n    $id: ID!\n    $saleInput: SaleInput!\n    $channelInput: SaleChannelListingInput!\n) {\n    saleUpdate(id: $id, input: $saleInput) {\n        errors {\n            code\n        }\n        sale {\n            channelListings {\n                id\n            }\n        }\n    }\n    saleChannelListingUpdate(id: $id, input: $channelInput) {\n        errors {\n            code\n        }\n        sale {\n            channelListings {\n                id\n                channel {\n                    id\n                }\n            }\n        }\n    }\n}\n'

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_invalidate_data_sale_channel_listings_update(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_USD):
    if False:
        while True:
            i = 10
    discount_value = 10
    promotion = promotion_converted_from_sale
    rule = promotion.rules.first()
    rule_name = rule.name
    rule.channels.remove(channel_USD)
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    variables = {'id': sale_id, 'saleInput': {}, 'channelInput': {'addChannels': [{'channelId': channel_id, 'discountValue': discount_value}]}}
    response = staff_api_client.post_graphql(SALE_AND_SALE_CHANNEL_LISTING_UPDATE_MUTATION, variables=variables, permissions=(permission_manage_discounts,))
    content = get_graphql_content(response)
    promotion.refresh_from_db()
    rules = promotion.rules.all()
    assert len(rules) == 2
    old_rule = rules.get(name=rule_name)
    new_rule = rules.get(name__isnull=True)
    assert new_rule.reward_value == discount_value
    assert old_rule.channels.first() is None
    assert new_rule.channels.first().id == channel_USD.id
    assert len(new_rule.channels.all()) == 1
    sale_errors = content['data']['saleUpdate']['errors']
    channel_listings_errors = content['data']['saleChannelListingUpdate']['errors']
    assert not sale_errors
    assert not channel_listings_errors
    sale_data = content['data']['saleUpdate']['sale']
    channel_listings_data = content['data']['saleChannelListingUpdate']['sale']
    assert sale_data['channelListings'] == []
    assert channel_listings_data['channelListings'][0]['channel']['id'] == channel_id
    mock_update_products_discounted_prices_of_promotion_task.delay.assert_called_once_with(promotion.pk)

@patch('saleor.graphql.discount.mutations.sale.sale_channel_listing_update.update_products_discounted_prices_of_promotion_task')
def test_sale_channel_listing_remove_all_channels_multiple_times(mock_update_products_discounted_prices_of_promotion_task, staff_api_client, promotion_converted_from_sale, permission_manage_discounts, channel_PLN, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    promotion = promotion_converted_from_sale
    sale_id = graphene.Node.to_global_id('Sale', promotion.old_sale_id)
    channel_usd_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    channel_pln_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    discounted = 2
    staff_api_client.user.user_permissions.add(permission_manage_discounts)
    query = SALE_CHANNEL_LISTING_UPDATE_MUTATION
    mock_update_products_discounted_prices_of_promotion_task.return_value = None
    variables_add = {'id': sale_id, 'input': {'addChannels': [{'channelId': channel_usd_id, 'discountValue': discounted}, {'channelId': channel_pln_id, 'discountValue': discounted}]}}
    variables_remove = {'id': sale_id, 'input': {'removeChannels': [channel_usd_id, channel_pln_id]}}
    staff_api_client.post_graphql(query, variables=variables_add)
    staff_api_client.post_graphql(query, variables=variables_remove)
    staff_api_client.post_graphql(query, variables=variables_add)
    staff_api_client.post_graphql(query, variables=variables_remove)
    promotion.refresh_from_db()
    rules = promotion.rules.all()
    assert len(rules) == 1
    assert not rules[0].channels.first()

def test_sale_channel_listing_update_not_found_error(staff_api_client, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    query = SALE_CHANNEL_LISTING_UPDATE_MUTATION
    variables = {'id': graphene.Node.to_global_id('Sale', '0'), 'input': {'removeChannels': []}}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert not content['data']['saleChannelListingUpdate']['sale']
    errors = content['data']['saleChannelListingUpdate']['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == DiscountErrorCode.NOT_FOUND.name