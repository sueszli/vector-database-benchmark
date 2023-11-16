from urllib.parse import parse_qs, urlparse
from django.urls import reverse
from rest_framework import status

def test_slack_oauth_flow_returns_401_if_secret_is_invalid(environment_api_key, api_client):
    if False:
        print('Hello World!')
    base_url = reverse('api-v1:environments:integrations-slack-slack-oauth-init', args=[environment_api_key])
    url = f'{base_url}?signature=not_a_signature'
    response = api_client.get(url)
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_slack_oauth_flow(mocker, settings, api_client, admin_client, environment_api_key, environment):
    if False:
        i = 10
        return i + 15
    settings.SLACK_CLIENT_ID = 'slack_id'
    settings.SLACK_CLIENT_SECRET = 'client_secret'
    redirect_url = 'http://localhost'
    url = reverse('api-v1:environments:integrations-slack-get-temporary-signature', args=[environment_api_key])
    signature = admin_client.get(url).json()['signature']
    base_url = reverse('api-v1:environments:integrations-slack-slack-oauth-init', args=[environment_api_key])
    url = f'{base_url}?redirect_url={redirect_url}&signature={signature}'
    response = api_client.get(url)
    assert response.status_code == status.HTTP_302_FOUND
    params = parse_qs(urlparse(response.url).query)
    state = params['state'][0]
    assert params['client_id'][0] == settings.SLACK_CLIENT_ID
    mocked_get_bot_token = mocker.MagicMock(return_value='bot_token')
    mocked_slack_wrapper = mocker.patch('integrations.slack.views.SlackWrapper')
    mocked_slack_wrapper.return_value.get_bot_token = mocked_get_bot_token
    callback_url = params['redirect_uri'][0]
    code = 'random_slack_code'
    response = admin_client.get(f'{callback_url}?state={state}&code={code}')
    assert response.status_code == status.HTTP_302_FOUND
    assert response.url == redirect_url
    mocked_slack_wrapper.assert_called_with()
    mocked_get_bot_token.assert_called_with(code, callback_url)

def test_slack_oauth_callback_returns_400_if_redirect_url_is_not_found_in_session(mocker, django_client, environment, environment_api_key, slack_bot_token):
    if False:
        print('Hello World!')
    url = reverse('api-v1:environments:integrations-slack-slack-oauth-callback', args=[environment_api_key])
    mocker.patch('integrations.slack.views.SlackWrapper.get_bot_token', return_value=slack_bot_token)
    mocker.patch('integrations.slack.views.validate_state', return_value=True)
    response = django_client.get(f'{url}?state=state&code=code')
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()['detail'] == 'Redirect URL not found in request session'

def test_slack_oauth_init_returns_401_for_user_that_does_not_have_access_to_the_environment(environment, environment_api_key, settings, django_user_model, api_client):
    if False:
        i = 10
        return i + 15
    a_non_admin_user = django_user_model.objects.create(username='random_user')
    api_client.force_authenticate(user=a_non_admin_user)
    url = reverse('api-v1:environments:integrations-slack-get-temporary-signature', args=[environment_api_key])
    signature = api_client.get(url).json()['signature']
    settings.SLACK_CLIENT_ID = 'slack_id'
    settings.SLACK_CLIENT_SECRET = 'client_secret'
    base_url = reverse('api-v1:environments:integrations-slack-slack-oauth-init', args=[environment_api_key])
    url = f'{base_url}?redirect_url=http://localhost&signature={signature}'
    response = api_client.get(url)
    assert response.status_code == status.HTTP_403_FORBIDDEN