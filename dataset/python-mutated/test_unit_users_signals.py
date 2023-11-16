from users.models import FFAdminUser
from users.signals import create_pipedrive_lead_signal

def test_create_pipedrive_lead_signal_calls_task_if_user_created(mocker, settings, django_user_model):
    if False:
        for i in range(10):
            print('nop')
    mocked_create_pipedrive_lead = mocker.patch('users.signals.create_pipedrive_lead')
    settings.ENABLE_PIPEDRIVE_LEAD_TRACKING = True
    user = django_user_model.objects.create(email='test@example.com')
    mocked_create_pipedrive_lead.delay.assert_called_once_with(args=(user.id,))

def test_create_pipedrive_lead_signal_does_not_call_task_if_user_not_created(mocker, settings):
    if False:
        return 10
    mocked_create_pipedrive_lead = mocker.patch('users.signals.create_pipedrive_lead')
    user = mocker.MagicMock()
    settings.PIPEDRIVE_API_TOKEN = 'some-token'
    create_pipedrive_lead_signal(FFAdminUser, instance=user, created=False)
    mocked_create_pipedrive_lead.delay.assert_not_called()

def test_create_pipedrive_lead_signal_does_not_call_task_if_pipedrive_not_configured(mocker, settings):
    if False:
        i = 10
        return i + 15
    mocked_create_pipedrive_lead = mocker.patch('users.signals.create_pipedrive_lead')
    user = mocker.MagicMock()
    settings.PIPEDRIVE_API_TOKEN = None
    create_pipedrive_lead_signal(FFAdminUser, instance=user, created=False)
    mocked_create_pipedrive_lead.delay.assert_not_called()