import os
from unittest import mock
from django.core.management import call_command
from ..models import Group, User

def test_createsuperuser_command(db):
    if False:
        for i in range(10):
            print('nop')
    call_command('createsuperuser', email='testAdmin@example.com', interactive=False)
    assert User.objects.count() == 1
    assert User.objects.get(email='testAdmin@example.com', is_staff=True, is_superuser=True)
    assert Group.objects.count() == 1
    assert Group.objects.get(name='Full Access')

def test_createsuperuser_command_dont_override_group(db):
    if False:
        for i in range(10):
            print('nop')
    group = Group.objects.create(name='Full Access')
    call_command('createsuperuser', email='testAdmin@example.com', interactive=False)
    assert User.objects.count() == 1
    assert User.objects.get(email='testAdmin@example.com', is_staff=True, is_superuser=True)
    group.refresh_from_db()
    assert not group.permissions.all()

@mock.patch.dict(os.environ, {'DJANGO_SUPERUSER_EMAIL': 'adminTest@example.com'})
def test_createsuperuser_command_email_from_settings(db):
    if False:
        for i in range(10):
            print('nop')
    call_command('createsuperuser', interactive=False)
    assert User.objects.count() == 1
    assert User.objects.get(email='adminTest@example.com', is_staff=True, is_superuser=True)
    assert Group.objects.count() == 1
    assert Group.objects.get(name='Full Access')