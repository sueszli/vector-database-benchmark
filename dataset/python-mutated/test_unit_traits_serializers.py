from core.constants import STRING
from environments.identities.traits.models import Trait
from environments.sdk.serializers import SDKBulkCreateUpdateTraitSerializer

def test_bulk_create_update_serializer_save_many(identity, django_assert_num_queries, mocker):
    if False:
        print('Hello World!')
    trait_key_to_update = 'foo'
    trait_value_to_update = 'bar'
    Trait.objects.create(identity=identity, trait_key=trait_key_to_update, string_value=trait_value_to_update, value_type=STRING)
    trait_key_to_delete = 'to-delete'
    Trait.objects.create(identity=identity, trait_key=trait_key_to_delete, value_type=STRING, string_value='irrelevant')
    identity_data = {'identifier': identity.identifier}
    updated_trait_value = f'{trait_value_to_update} updated'
    data = [{'trait_key': 'new-trait-1', 'trait_value': 'foo', 'identity': identity_data}, {'trait_key': 'new-trait-2', 'trait_value': 'foo', 'identity': identity_data}, {'trait_key': trait_key_to_update, 'trait_value': updated_trait_value, 'identity': identity_data}, {'trait_key': trait_key_to_delete, 'trait_value': None, 'identity': identity_data}]
    mocked_request = mocker.MagicMock(environment=identity.environment)
    with django_assert_num_queries(6):
        serializer = SDKBulkCreateUpdateTraitSerializer(data=data, many=True, context={'environment': identity.environment, 'request': mocked_request})
        serializer.is_valid(raise_exception=True)
        serializer.save()
    assert identity.identity_traits.count() == 3
    assert identity.identity_traits.get(trait_key=trait_key_to_update).trait_value == updated_trait_value