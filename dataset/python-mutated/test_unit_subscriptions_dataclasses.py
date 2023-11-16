import pytest
from organisations.subscriptions.metadata import BaseSubscriptionMetadata

class SourceASubscriptionMetadata(BaseSubscriptionMetadata):
    payment_source = 'SOURCE_A'

class SourceBSubscriptionMetadata(BaseSubscriptionMetadata):
    payment_source = 'SOURCE_B'

def test_base_subscription_metadata_add_raises_error_if_not_matching_payment_source():
    if False:
        i = 10
        return i + 15
    source_a_metadata = SourceASubscriptionMetadata(seats=1, api_calls=50000)
    source_b_metadata = SourceBSubscriptionMetadata(seats=1, api_calls=50000)
    with pytest.raises(TypeError):
        source_a_metadata + source_b_metadata

@pytest.mark.parametrize('add_to, add, expected_result', ((SourceASubscriptionMetadata(seats=1, api_calls=50000), SourceASubscriptionMetadata(seats=1, api_calls=50000), SourceASubscriptionMetadata(seats=2, api_calls=100000, projects=None)), (SourceASubscriptionMetadata(seats=1, api_calls=50000, projects=1), SourceASubscriptionMetadata(seats=1, api_calls=50000, projects=1), SourceASubscriptionMetadata(seats=2, api_calls=100000, projects=2)), (SourceASubscriptionMetadata(seats=1, api_calls=50000, projects=1), SourceASubscriptionMetadata(seats=1, api_calls=50000), SourceASubscriptionMetadata(seats=2, api_calls=100000, projects=1)), (SourceASubscriptionMetadata(seats=1, api_calls=50000), SourceASubscriptionMetadata(seats=1, api_calls=50000, projects=1), SourceASubscriptionMetadata(seats=2, api_calls=100000, projects=1))))
def test_base_subscription_metadata_add(add_to, add, expected_result):
    if False:
        i = 10
        return i + 15
    assert add_to + add == expected_result