import json
import urllib
from django.urls import reverse
from rest_framework import status
from rest_framework.exceptions import NotFound
from edge_api.identities.views import EdgeIdentityViewSet

def test_get_identities_returns_bad_request_if_dynamo_is_not_enabled(admin_client, environment, environment_api_key):
    if False:
        i = 10
        return i + 15
    url = reverse('api-v1:environments:environment-edge-identities-list', args=[environment_api_key])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_get_identity(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, edge_identity_dynamo_wrapper_mock):
    if False:
        for i in range(10):
            print('nop')
    identity_uuid = identity_document['identity_uuid']
    url = reverse('api-v1:environments:environment-edge-identities-detail', args=[environment_api_key, identity_uuid])
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.return_value = identity_document
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['identity_uuid'] == identity_uuid
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.assert_called_with(identity_uuid)

def test_get_identity_returns_404_if_identity_does_not_exists(admin_client, dynamo_enabled_environment, environment_api_key, edge_identity_dynamo_wrapper_mock):
    if False:
        return 10
    url = reverse('api-v1:environments:environment-edge-identities-detail', args=[environment_api_key, 'identity_uuid_that_does_not_exists'])
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.side_effect = NotFound
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_create_identity(admin_client, dynamo_enabled_environment, environment_api_key, edge_identity_dynamo_wrapper_mock, identity_document):
    if False:
        i = 10
        return i + 15
    identifier = identity_document['identifier']
    composite_key = identity_document['composite_key']
    url = reverse('api-v1:environments:environment-edge-identities-list', args=[environment_api_key])
    edge_identity_dynamo_wrapper_mock.get_item.return_value = None
    response = admin_client.post(url, data={'identifier': identifier})
    edge_identity_dynamo_wrapper_mock.get_item.assert_called_with(composite_key)
    (name, args, _) = edge_identity_dynamo_wrapper_mock.mock_calls[1]
    assert name == 'put_item'
    assert args[0]['identifier'] == identifier
    assert args[0]['composite_key'] == composite_key
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()['identifier'] == identifier
    assert response.json()['identity_uuid'] is not None

def test_create_identity_returns_400_if_identity_already_exists(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, edge_identity_dynamo_wrapper_mock):
    if False:
        i = 10
        return i + 15
    identifier = identity_document['identifier']
    url = reverse('api-v1:environments:environment-edge-identities-list', args=[environment_api_key])
    edge_identity_dynamo_wrapper_mock.get_item.return_value = identity_document
    response = admin_client.post(url, data={'identifier': identifier})
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_delete_identity(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, edge_identity_dynamo_wrapper_mock):
    if False:
        return 10
    identity_uuid = identity_document['identity_uuid']
    url = reverse('api-v1:environments:environment-edge-identities-detail', args=[environment_api_key, identity_uuid])
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.return_value = identity_document
    response = admin_client.delete(url)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.assert_called_with(identity_uuid)
    edge_identity_dynamo_wrapper_mock.delete_item.assert_called_with(identity_document['composite_key'])

def test_identity_list_pagination(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, edge_identity_dynamo_wrapper_mock):
    if False:
        for i in range(10):
            print('nop')
    identity_item_key = {k: v for (k, v) in identity_document.items() if k in ['composite_key', 'environment_api_key', 'identifier']}
    base_url = reverse('api-v1:environments:environment-edge-identities-list', args=[environment_api_key])
    url = f'{base_url}?page_size=1'
    edge_identity_dynamo_wrapper_mock.get_all_items.side_effect = [{'Items': [identity_document], 'Count': 1, 'LastEvaluatedKey': identity_item_key}, {'Items': [identity_document], 'Count': 1, 'LastEvaluatedKey': None}]
    response = admin_client.get(url)
    assert response.status_code == 200
    response = response.json()
    last_evaluated_key = response['last_evaluated_key']
    url = f'{url}&last_evaluated_key={last_evaluated_key}'
    response = admin_client.get(url)
    edge_identity_dynamo_wrapper_mock.get_all_items.assert_called_with(environment_api_key, 1, identity_item_key)
    assert response.status_code == 200
    assert response.json()['last_evaluated_key'] is None

def test_get_identities_list(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, edge_identity_dynamo_wrapper_mock):
    if False:
        print('Hello World!')
    url = reverse('api-v1:environments:environment-edge-identities-list', args=[environment_api_key])
    edge_identity_dynamo_wrapper_mock.get_all_items.return_value = {'Items': [identity_document], 'Count': 1}
    response = admin_client.get(url)
    assert response.json()['results'][0]['identifier'] == identity_document['identifier']
    assert len(response.json()['results']) == 1
    assert response.status_code == status.HTTP_200_OK
    edge_identity_dynamo_wrapper_mock.get_all_items.assert_called_with(environment_api_key, 999, None)

def test_search_identities_without_exact_match(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, edge_identity_dynamo_wrapper_mock):
    if False:
        for i in range(10):
            print('nop')
    identifier = identity_document['identifier']
    base_url = reverse('api-v1:environments:environment-edge-identities-list', args=[environment_api_key])
    url = '%s?q=%s' % (base_url, identifier)
    edge_identity_dynamo_wrapper_mock.search_items_with_identifier.return_value = {'Items': [identity_document], 'Count': 1}
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['results'][0]['identifier'] == identifier
    assert len(response.json()['results']) == 1
    edge_identity_dynamo_wrapper_mock.search_items_with_identifier.assert_called_with(environment_api_key, identifier, EdgeIdentityViewSet.dynamo_identifier_search_functions['BEGINS_WITH'], 999, None)

def test_search_for_identities_with_exact_match(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, edge_identity_dynamo_wrapper_mock):
    if False:
        print('Hello World!')
    identifier = identity_document['identifier']
    base_url = reverse('api-v1:environments:environment-edge-identities-list', args=[environment_api_key])
    url = '%s?%s' % (base_url, urllib.parse.urlencode({'q': f'"{identifier}"'}))
    edge_identity_dynamo_wrapper_mock.search_items_with_identifier.return_value = {'Items': [identity_document], 'Count': 1}
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['results'][0]['identifier'] == identifier
    assert len(response.json()['results']) == 1
    edge_identity_dynamo_wrapper_mock.search_items_with_identifier.assert_called_with(environment_api_key, identifier, EdgeIdentityViewSet.dynamo_identifier_search_functions['EQUAL'], 999, None)

def test_edge_identities_traits_list(admin_client, environment_api_key, identity_document, identity_traits, dynamo_enabled_environment, edge_identity_dynamo_wrapper_mock):
    if False:
        print('Hello World!')
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.return_value = identity_document
    identity_uuid = identity_document['identity_uuid']
    url = reverse('api-v1:environments:environment-edge-identities-get-traits', args=[environment_api_key, identity_uuid])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == identity_traits
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.assert_called_with(identity_uuid)

def test_edge_identities_trait_delete(admin_client, environment_api_key, dynamo_enabled_environment, identity_document, identity_traits, edge_identity_dynamo_wrapper_mock):
    if False:
        while True:
            i = 10
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.return_value = identity_document
    identity_uuid = identity_document['identity_uuid']
    trait_key = identity_traits[0]['trait_key']
    url = reverse('api-v1:environments:environment-edge-identities-update-traits', args=[environment_api_key, identity_uuid])
    data = {'trait_key': trait_key, 'trait_value': None}
    response = admin_client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.assert_called_with(identity_uuid)
    (name, args, _) = edge_identity_dynamo_wrapper_mock.mock_calls[1]
    assert name == 'put_item'
    assert not list(filter(lambda trait: trait['trait_key'] == trait_key, args[0]['identity_traits']))

def test_edge_identities_create_trait(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, edge_identity_dynamo_wrapper_mock):
    if False:
        return 10
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.return_value = identity_document
    identity_uuid = identity_document['identity_uuid']
    url = reverse('api-v1:environments:environment-edge-identities-update-traits', args=[environment_api_key, identity_uuid])
    trait_key = 'new_trait_key'
    data = {'trait_key': trait_key, 'trait_value': 'new_trait_value'}
    response = admin_client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['trait_key'] == trait_key
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.assert_called_with(identity_uuid)
    (name, args, _) = edge_identity_dynamo_wrapper_mock.mock_calls[1]
    assert name == 'put_item'
    assert list(filter(lambda trait: trait['trait_key'] == trait_key, args[0]['identity_traits']))

def test_edge_identities_update_trait(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, identity_traits, edge_identity_dynamo_wrapper_mock, mocker):
    if False:
        for i in range(10):
            print('nop')
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.return_value = identity_document
    identity_uuid = identity_document['identity_uuid']
    trait_key = identity_traits[0]['trait_key']
    url = reverse('api-v1:environments:environment-edge-identities-update-traits', args=[environment_api_key, identity_uuid])
    updated_trait_value = 'updated_trait_value'
    data = {'trait_key': trait_key, 'trait_value': updated_trait_value}
    response = admin_client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['trait_key'] == trait_key
    assert response.json()['trait_value'] == updated_trait_value
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.assert_called_with(identity_uuid)
    (name, args, _) = edge_identity_dynamo_wrapper_mock.mock_calls[1]
    assert name == 'put_item'
    assert list(filter(lambda trait: trait['trait_key'] == trait_key and trait['trait_value'] == updated_trait_value, args[0]['identity_traits']))

def test_edge_identities_update_trait_with_same_value(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, identity_traits, edge_identity_dynamo_wrapper_mock, mocker):
    if False:
        return 10
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.return_value = identity_document
    identity_uuid = identity_document['identity_uuid']
    trait_key = identity_traits[0]['trait_key']
    trait_value = identity_traits[0]['trait_value']
    url = reverse('api-v1:environments:environment-edge-identities-update-traits', args=[environment_api_key, identity_uuid])
    data = {'trait_key': trait_key, 'trait_value': trait_value}
    response = admin_client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_200_OK
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.assert_called_with(identity_uuid)
    edge_identity_dynamo_wrapper_mock.put_item.assert_not_called()

def test_edge_identities_update_traits_returns_400_if_persist_trait_data_is_false(admin_client, dynamo_enabled_environment, environment_api_key, identity_document, identity_traits, edge_identity_dynamo_wrapper_mock, organisation_with_persist_trait_data_disabled):
    if False:
        while True:
            i = 10
    edge_identity_dynamo_wrapper_mock.get_item_from_uuid_or_404.return_value = identity_document
    identity_uuid = identity_document['identity_uuid']
    trait_key = identity_traits[0]['trait_key']
    url = reverse('api-v1:environments:environment-edge-identities-update-traits', args=[environment_api_key, identity_uuid])
    updated_trait_value = 'updated_trait_value'
    data = {'trait_key': trait_key, 'trait_value': updated_trait_value}
    response = admin_client.put(url, data=json.dumps(data), content_type='application/json')
    assert response.status_code == status.HTTP_400_BAD_REQUEST