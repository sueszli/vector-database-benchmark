import pytest
from slack_sdk.errors import SlackApiError
from audit.models import AuditLog
from environments.models import Environment
from integrations.slack.exceptions import SlackChannelJoinError
from integrations.slack.slack import SlackChannel, SlackWrapper

def test_get_channels_data_response_structure(mocker, mocked_slack_internal_client):
    if False:
        for i in range(10):
            print('nop')
    api_token = 'test_token'
    cursor = 'dGVhbTpDMDI3MEpNRldNVg=='
    response_data = {'ok': True, 'channels': [{'id': 'id1', 'name': 'channel1', 'is_channel': True, 'num_members': 3}, {'id': 'id2', 'name': 'channel2', 'is_channel': True, 'num_members': 3}], 'response_metadata': {'next_cursor': cursor}}
    some_kwargs = {'key': 'value'}
    mocked_slack_internal_client.conversations_list.return_value = response_data
    channels_data = SlackWrapper(api_token=api_token).get_channels_data(**some_kwargs)
    assert channels_data.channels == [SlackChannel('channel1', 'id1'), SlackChannel('channel2', 'id2')]
    assert channels_data.cursor == cursor
    mocked_slack_internal_client.conversations_list.assert_called_with(exclude_archived=True, **some_kwargs)

def test_client_makes_correct_calls(mocker):
    if False:
        i = 10
        return i + 15
    api_token = 'random_token'
    mocked_web_client = mocker.patch('integrations.slack.slack.WebClient')
    slack_wrapper = SlackWrapper(api_token=api_token)
    assert mocked_web_client.return_value == slack_wrapper._client
    mocked_web_client.assert_called_with(token=api_token)

def test_join_channel_makes_correct_call(mocker, mocked_slack_internal_client):
    if False:
        return 10
    channel = 'channel_1'
    api_token = 'random_token'
    SlackWrapper(api_token=api_token, channel_id=channel).join_channel()
    mocked_slack_internal_client.conversations_join.assert_called_with(channel=channel)

def test_join_channel_raises_slack_channel_join_error_on_slack_api_error(mocker, mocked_slack_internal_client):
    if False:
        for i in range(10):
            print('nop')
    channel = 'channel_1'
    api_token = 'random_token'
    mocked_slack_internal_client.conversations_join.side_effect = SlackApiError(message='server_error', response={'error': 'some_error_code'})
    with pytest.raises(SlackChannelJoinError):
        SlackWrapper(api_token=api_token, channel_id=channel).join_channel()

def test_get_bot_token_makes_correct_calls(mocker, settings, mocked_slack_internal_client):
    if False:
        print('Hello World!')
    code = 'test_code'
    redirect_uri = 'http://localhost'
    settings.SLACK_CLIENT_ID = 'test_client_id'
    settings.SLACK_CLIENT_SECRET = 'test_client_secret'
    slack_wrapper = SlackWrapper()
    token = slack_wrapper.get_bot_token(code, redirect_uri)
    mocked_slack_internal_client.oauth_v2_access.assert_called_with(client_id=settings.SLACK_CLIENT_ID, client_secret=settings.SLACK_CLIENT_SECRET, code=code, redirect_uri=redirect_uri)
    assert token == mocked_slack_internal_client.oauth_v2_access.return_value.get.return_value

def test_slack_initialized_correctly(mocker, mocked_slack_internal_client):
    if False:
        print('Hello World!')
    api_token = 'test_token'
    channel_id = 'channel_id_1'
    slack_wrapper = SlackWrapper(api_token, channel_id)
    assert slack_wrapper.channel_id == channel_id
    assert slack_wrapper._client == mocked_slack_internal_client

def test_track_event_makes_correct_call(mocked_slack_internal_client):
    if False:
        return 10
    api_token = 'test_token'
    channel_id = 'channel_id_1'
    event = {'blocks': []}
    slack_wrapper = SlackWrapper(api_token, channel_id)
    slack_wrapper._track_event(event)
    mocked_slack_internal_client.chat_postMessage.assert_called_with(channel=channel_id, blocks=event['blocks'])

def test_slack_generate_event_data_with_correct_values(django_user_model):
    if False:
        return 10
    log = 'some log data'
    author = django_user_model(email='test@email.com')
    environment = Environment(name='test')
    audit_log_record = AuditLog(log=log, author=author, environment=environment)
    event_data = SlackWrapper.generate_event_data(audit_log_record=audit_log_record)
    assert event_data['blocks'] == [{'type': 'section', 'text': {'type': 'plain_text', 'text': log}}, {'type': 'section', 'fields': [{'type': 'mrkdwn', 'text': f'*Environment:*\n{environment.name}'}, {'type': 'mrkdwn', 'text': f'*User:*\n{author.email}'}]}]