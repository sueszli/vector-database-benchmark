from collections import defaultdict
import graphene
import pytest
from django.core.exceptions import ValidationError
from .....shipping.error_codes import ShippingErrorCode
from ...mutations import BaseChannelListingMutation

def test_validate_duplicated_channel_ids(channel_PLN, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    second_channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    errors = defaultdict(list)
    result = BaseChannelListingMutation.validate_duplicated_channel_ids([channel_id], [second_channel_id], errors, ShippingErrorCode.DUPLICATED_INPUT_ITEM.value)
    assert result is None
    assert errors['input'] == []

def test_validate_duplicated_channel_ids_with_duplicates(channel_PLN):
    if False:
        i = 10
        return i + 15
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    second_channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    error_code = ShippingErrorCode.DUPLICATED_INPUT_ITEM.value
    errors = defaultdict(list)
    result = BaseChannelListingMutation.validate_duplicated_channel_ids([channel_id], [second_channel_id], errors, error_code)
    assert result is None
    assert errors['input'][0].code == error_code

def test_validate_duplicated_channel_values(channel_PLN, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    second_channel_id = graphene.Node.to_global_id('Channel', channel_USD.id)
    error_code = ShippingErrorCode.DUPLICATED_INPUT_ITEM.value
    errors = defaultdict(list)
    field = 'add_channels'
    result = BaseChannelListingMutation.validate_duplicated_channel_values([channel_id, second_channel_id], field, errors, error_code)
    assert result is None
    assert errors[field] == []

def test_validate_duplicated_channel_values_with_duplicates(channel_PLN):
    if False:
        print('Hello World!')
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    second_channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    error_code = ShippingErrorCode.DUPLICATED_INPUT_ITEM.value
    errors = defaultdict(list)
    field = 'add_channels'
    result = BaseChannelListingMutation.validate_duplicated_channel_values([channel_id, second_channel_id], field, errors, error_code)
    assert result is None
    assert errors[field][0].code == error_code

def test_clean_channels_add_channels(channel_PLN):
    if False:
        for i in range(10):
            print('nop')
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    error_code = ShippingErrorCode.DUPLICATED_INPUT_ITEM.value
    errors = defaultdict(list)
    result = BaseChannelListingMutation.clean_channels(None, {'add_channels': [{'channel_id': channel_id}]}, errors, error_code)
    assert result == {'add_channels': [{'channel_id': channel_id, 'channel': channel_PLN}], 'remove_channels': []}
    assert errors['input'] == []

def test_clean_channels_remove_channels(channel_PLN):
    if False:
        return 10
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    error_code = ShippingErrorCode.DUPLICATED_INPUT_ITEM.value
    errors = defaultdict(list)
    result = BaseChannelListingMutation.clean_channels(None, {'remove_channels': [channel_id]}, errors, error_code)
    assert result == {'add_channels': [], 'remove_channels': [str(channel_PLN.id)]}
    assert errors['input'] == []

def test_clean_channels_remove_channels_is_null(channel_PLN):
    if False:
        print('Hello World!')
    channel_id = None
    error_code = ShippingErrorCode.DUPLICATED_INPUT_ITEM.value
    errors = defaultdict(list)
    result = BaseChannelListingMutation.clean_channels(None, {'remove_channels': channel_id}, errors, error_code)
    assert result == {'add_channels': [], 'remove_channels': []}
    assert errors['input'] == []

def test_test_clean_channels_with_errors(channel_PLN):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Channel', channel_PLN.id)
    error_code = ShippingErrorCode.DUPLICATED_INPUT_ITEM.value
    errors = defaultdict(list)
    result = BaseChannelListingMutation.clean_channels(None, {'remove_channels': [channel_id, channel_id]}, errors, error_code)
    assert result == {}
    assert errors['remove_channels'][0].code == error_code

def test_test_clean_channels_invalid_object_type(channel_PLN):
    if False:
        while True:
            i = 10
    channel_id = graphene.Node.to_global_id('Product', channel_PLN.id)
    error_code = ShippingErrorCode.GRAPHQL_ERROR.value
    errors = defaultdict(list)
    with pytest.raises(ValidationError) as error:
        BaseChannelListingMutation.clean_channels(None, {'remove_channels': [channel_id]}, errors, error_code)
    assert error.value.error_dict['remove_channels'][0].message == f'Must receive Channel id: {channel_id}.'