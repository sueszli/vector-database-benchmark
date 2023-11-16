import datetime
from unittest.mock import MagicMock, patch
from django.core.files import File
from django.utils import timezone
from freezegun import freeze_time
from ......account.models import User
from ......core.tokens import account_delete_token_generator
from ......thumbnail.models import Thumbnail
from .....tests.utils import assert_no_permission, get_graphql_content
ACCOUNT_DELETE_MUTATION = '\n    mutation AccountDelete($token: String!){\n        accountDelete(token: $token){\n            errors{\n                field\n                message\n            }\n        }\n    }\n'

@patch('saleor.core.tasks.delete_from_storage_task.delay')
@patch('saleor.plugins.manager.PluginsManager.account_deleted')
@freeze_time('2018-05-31 12:00:01')
def test_account_delete(mocked_account_deleted, delete_from_storage_task_mock, user_api_client, media_root):
    if False:
        print('Hello World!')
    thumbnail_mock = MagicMock(spec=File)
    thumbnail_mock.name = 'image.jpg'
    user = user_api_client.user
    user.last_login = timezone.now()
    user.save(update_fields=['last_login'])
    user_id = user.id
    thumbnail = Thumbnail.objects.create(user=user, size=128, image=thumbnail_mock)
    assert user.thumbnails.all()
    img_path = thumbnail.image.name
    token = account_delete_token_generator.make_token(user)
    variables = {'token': token}
    response = user_api_client.post_graphql(ACCOUNT_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['accountDelete']
    assert not data['errors']
    assert not User.objects.filter(pk=user.id).exists()
    assert not Thumbnail.objects.filter(user_id=user_id).exists()
    delete_from_storage_task_mock.assert_called_once_with(img_path)
    mocked_account_deleted.assert_called_once_with(user)

@freeze_time('2018-05-31 12:00:01')
def test_account_delete_user_never_log_in(user_api_client):
    if False:
        print('Hello World!')
    user = user_api_client.user
    token = account_delete_token_generator.make_token(user)
    variables = {'token': token}
    response = user_api_client.post_graphql(ACCOUNT_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['accountDelete']
    assert not data['errors']
    assert not User.objects.filter(pk=user.id).exists()

@freeze_time('2018-05-31 12:00:01')
def test_account_delete_log_out_after_deletion_request(user_api_client):
    if False:
        i = 10
        return i + 15
    user = user_api_client.user
    user.last_login = timezone.now()
    user.save(update_fields=['last_login'])
    token = account_delete_token_generator.make_token(user)
    user.last_login = timezone.now() + datetime.timedelta(hours=1)
    user.save(update_fields=['last_login'])
    variables = {'token': token}
    response = user_api_client.post_graphql(ACCOUNT_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['accountDelete']
    assert not data['errors']
    assert not User.objects.filter(pk=user.id).exists()

def test_account_delete_invalid_token(user_api_client):
    if False:
        i = 10
        return i + 15
    user = user_api_client.user
    variables = {'token': 'invalid'}
    response = user_api_client.post_graphql(ACCOUNT_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['accountDelete']
    assert len(data['errors']) == 1
    assert data['errors'][0]['message'] == 'Invalid or expired token.'
    assert User.objects.filter(pk=user.id).exists()

def test_account_delete_anonymous_user(api_client):
    if False:
        while True:
            i = 10
    variables = {'token': 'invalid'}
    response = api_client.post_graphql(ACCOUNT_DELETE_MUTATION, variables)
    assert_no_permission(response)

def test_account_delete_staff_user(staff_api_client):
    if False:
        return 10
    user = staff_api_client.user
    variables = {'token': 'invalid'}
    response = staff_api_client.post_graphql(ACCOUNT_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['accountDelete']
    assert len(data['errors']) == 1
    assert data['errors'][0]['message'] == 'Cannot delete a staff account.'
    assert User.objects.filter(pk=user.id).exists()

@freeze_time('2018-05-31 12:00:01')
def test_account_delete_other_customer_token(user_api_client):
    if False:
        print('Hello World!')
    user = user_api_client.user
    other_user = User.objects.create(email='temp@example.com')
    token = account_delete_token_generator.make_token(other_user)
    variables = {'token': token}
    response = user_api_client.post_graphql(ACCOUNT_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    data = content['data']['accountDelete']
    assert len(data['errors']) == 1
    assert data['errors'][0]['message'] == 'Invalid or expired token.'
    assert User.objects.filter(pk=user.id).exists()
    assert User.objects.filter(pk=other_user.id).exists()

@patch('saleor.plugins.webhook.plugin.trigger_webhooks_async')
@freeze_time('2018-05-31 12:00:01')
def test_account_delete_webhook_event_triggered(mocked_trigger_webhooks_async, settings, user_api_client, subscription_account_deleted_webhook):
    if False:
        while True:
            i = 10
    settings.PLUGINS = ['saleor.plugins.webhook.plugin.WebhookPlugin']
    user = user_api_client.user
    user.last_login = timezone.now()
    user.save(update_fields=['last_login'])
    token = account_delete_token_generator.make_token(user)
    variables = {'token': token}
    user_api_client.post_graphql(ACCOUNT_DELETE_MUTATION, variables)
    assert not User.objects.filter(pk=user.id).exists()
    mocked_trigger_webhooks_async.assert_called()