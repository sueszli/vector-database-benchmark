import json
from cryptography.fernet import InvalidToken
from django.test.utils import override_settings
from django.conf import settings
from django.core.management import call_command
import os
import pytest
from awx.main import models
from awx.conf.models import Setting
from awx.main.management.commands import regenerate_secret_key
from awx.main.utils.encryption import encrypt_field, decrypt_field, encrypt_value
PREFIX = '$encrypted$UTF8$AESCBC$'

@pytest.mark.django_db
class TestKeyRegeneration:

    def test_encrypted_ssh_password(self, credential):
        if False:
            for i in range(10):
                print('nop')
        assert credential.inputs['password'].startswith(PREFIX)
        assert credential.get_input('password') == 'secret'
        new_key = regenerate_secret_key.Command().handle()
        new_cred = models.Credential.objects.get(pk=credential.pk)
        assert credential.inputs['password'] != new_cred.inputs['password']
        with pytest.raises(InvalidToken):
            new_cred.get_input('password')
        with override_settings(SECRET_KEY=new_key):
            assert new_cred.get_input('password') == 'secret'

    def test_encrypted_setting_values(self):
        if False:
            while True:
                i = 10
        settings.REDHAT_PASSWORD = 'sensitive'
        s = Setting.objects.filter(key='REDHAT_PASSWORD').first()
        assert s.value.startswith(PREFIX)
        assert settings.REDHAT_PASSWORD == 'sensitive'
        new_key = regenerate_secret_key.Command().handle()
        new_setting = Setting.objects.filter(key='REDHAT_PASSWORD').first()
        assert s.value != new_setting.value
        settings.cache.delete('REDHAT_PASSWORD')
        settings._awx_conf_memoizedcache.clear()
        with pytest.raises(InvalidToken):
            settings.REDHAT_PASSWORD
        settings._awx_conf_memoizedcache.clear()
        with override_settings(SECRET_KEY=new_key):
            assert settings.REDHAT_PASSWORD == 'sensitive'

    def test_encrypted_notification_secrets(self, notification_template_with_encrypt):
        if False:
            i = 10
            return i + 15
        nt = notification_template_with_encrypt
        nc = nt.notification_configuration
        assert nc['token'].startswith(PREFIX)
        Slack = nt.CLASS_FOR_NOTIFICATION_TYPE[nt.notification_type]

        class TestBackend(Slack):

            def __init__(self, *args, **kw):
                if False:
                    while True:
                        i = 10
                assert kw['token'] == 'token'

            def send_messages(self, messages):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        nt.CLASS_FOR_NOTIFICATION_TYPE['test'] = TestBackend
        nt.notification_type = 'test'
        nt.send('Subject', 'Body')
        new_key = regenerate_secret_key.Command().handle()
        new_nt = models.NotificationTemplate.objects.get(pk=nt.pk)
        assert nt.notification_configuration['token'] != new_nt.notification_configuration['token']
        with pytest.raises(InvalidToken):
            new_nt.CLASS_FOR_NOTIFICATION_TYPE['test'] = TestBackend
            new_nt.notification_type = 'test'
            new_nt.send('Subject', 'Body')
        with override_settings(SECRET_KEY=new_key):
            new_nt.send('Subject', 'Body')

    def test_job_start_args(self, job_factory):
        if False:
            while True:
                i = 10
        job = job_factory()
        job.start_args = json.dumps({'foo': 'bar'})
        job.start_args = encrypt_field(job, field_name='start_args')
        job.save()
        assert job.start_args.startswith(PREFIX)
        new_key = regenerate_secret_key.Command().handle()
        new_job = models.Job.objects.get(pk=job.pk)
        assert new_job.start_args != job.start_args
        with pytest.raises(InvalidToken):
            decrypt_field(new_job, field_name='start_args')
        with override_settings(SECRET_KEY=new_key):
            assert json.loads(decrypt_field(new_job, field_name='start_args')) == {'foo': 'bar'}

    @pytest.mark.parametrize('cls', ('JobTemplate', 'WorkflowJobTemplate'))
    def test_survey_spec(self, inventory, project, survey_spec_factory, cls):
        if False:
            for i in range(10):
                print('nop')
        params = {}
        if cls == 'JobTemplate':
            params['inventory'] = inventory
            params['project'] = project
        jt = getattr(models, cls).objects.create(name='Example Template', survey_spec=survey_spec_factory([{'variable': 'secret_key', 'default': encrypt_value('donttell', pk=None), 'type': 'password'}]), survey_enabled=True, **params)
        job = jt.create_unified_job()
        assert jt.survey_spec['spec'][0]['default'].startswith(PREFIX)
        assert job.survey_passwords == {'secret_key': '$encrypted$'}
        assert json.loads(job.decrypted_extra_vars())['secret_key'] == 'donttell'
        new_key = regenerate_secret_key.Command().handle()
        new_job = models.UnifiedJob.objects.get(pk=job.pk)
        assert new_job.extra_vars != job.extra_vars
        with pytest.raises(InvalidToken):
            new_job.decrypted_extra_vars()
        with override_settings(SECRET_KEY=new_key):
            assert json.loads(new_job.decrypted_extra_vars())['secret_key'] == 'donttell'

    def test_oauth2_application_client_secret(self, oauth_application):
        if False:
            print('Hello World!')
        secret = oauth_application.client_secret
        assert len(secret) == 128
        new_key = regenerate_secret_key.Command().handle()
        with pytest.raises(InvalidToken):
            models.OAuth2Application.objects.get(pk=oauth_application.pk).client_secret
        with override_settings(SECRET_KEY=new_key):
            assert models.OAuth2Application.objects.get(pk=oauth_application.pk).client_secret == secret

    def test_use_custom_key_with_tower_secret_key_env_var(self):
        if False:
            for i in range(10):
                print('nop')
        custom_key = 'MXSq9uqcwezBOChl/UfmbW1k4op+bC+FQtwPqgJ1u9XV'
        os.environ['TOWER_SECRET_KEY'] = custom_key
        new_key = call_command('regenerate_secret_key', '--use-custom-key')
        assert custom_key == new_key

    def test_use_custom_key_with_empty_tower_secret_key_env_var(self):
        if False:
            while True:
                i = 10
        os.environ['TOWER_SECRET_KEY'] = ''
        with pytest.raises(SystemExit) as e:
            call_command('regenerate_secret_key', '--use-custom-key')
        assert e.type == SystemExit
        assert e.value.code == 1

    def test_use_custom_key_with_no_tower_secret_key_env_var(self):
        if False:
            while True:
                i = 10
        os.environ.pop('TOWER_SECRET_KEY', None)
        with pytest.raises(SystemExit) as e:
            call_command('regenerate_secret_key', '--use-custom-key')
        assert e.type == SystemExit
        assert e.value.code == 1

    def test_with_tower_secret_key_env_var(self):
        if False:
            for i in range(10):
                print('nop')
        custom_key = 'MXSq9uqcwezBOChl/UfmbW1k4op+bC+FQtwPqgJ1u9XV'
        os.environ['TOWER_SECRET_KEY'] = custom_key
        new_key = call_command('regenerate_secret_key')
        assert custom_key != new_key