from dataclasses import asdict
from django.urls import reverse
from rest_framework import status

def test_get_channels_returns_400_when_slack_project_config_does_not_exist(admin_client, environment, environment_api_key):
    if False:
        for i in range(10):
            print('nop')
    url = reverse('api-v1:environments:integrations-slack-channels-list', args=[environment_api_key])
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert 'Slack api token not found. Please generate the token using oauth' == response.json()['detail']

def test_get_channels_pagination(mocker, admin_client, environment_api_key, slack_project_config, slack_channels_data_response, slack_bot_token, mocked_slack_wrapper):
    if False:
        i = 10
        return i + 15
    page_size = 10
    base_url = reverse('api-v1:environments:integrations-slack-channels-list', args=[environment_api_key])
    mocked_get_channels_data = mocker.MagicMock(return_value=slack_channels_data_response)
    mocked_slack_wrapper.return_value.get_channels_data = mocked_get_channels_data
    url = f'{base_url}?limit={page_size}'
    response = admin_client.get(url)
    mocked_get_channels_data.assert_called_with(limit=page_size)
    mocked_slack_wrapper.assert_called_with(api_token=slack_bot_token)
    assert len(response.json()['channels']) == len(slack_channels_data_response.channels)
    cursor = response.json()['cursor']
    url = f'{base_url}?cursor={cursor}&limit={page_size}'
    response = admin_client.get(url)
    mocked_get_channels_data.assert_called_with(cursor=cursor, limit=page_size)
    mocked_slack_wrapper.assert_called_with(api_token=slack_bot_token)

def test_get_channels_response_structure(mocker, admin_client, environment_api_key, slack_project_config, slack_channels_data_response, mocked_slack_wrapper):
    if False:
        print('Hello World!')
    url = reverse('api-v1:environments:integrations-slack-channels-list', args=[environment_api_key])
    mocked_get_channels_data = mocker.MagicMock(return_value=slack_channels_data_response)
    mocked_slack_wrapper.return_value.get_channels_data = mocked_get_channels_data
    response = admin_client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()['cursor'] == slack_channels_data_response.cursor
    assert response.json()['channels'][0] == asdict(slack_channels_data_response.channels[0])
    assert response.json()['channels'][1] == asdict(slack_channels_data_response.channels[1])