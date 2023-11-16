from email.headerregistry import Address
from typing import Any
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def host_for_subdomain(subdomain: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    if subdomain == '':
        return settings.EXTERNAL_HOST
    default_host = f'{subdomain}.{settings.EXTERNAL_HOST}'
    return settings.REALM_HOSTS.get(subdomain, default_host)

def get_fake_email_domain(realm: Any) -> str:
    if False:
        print('Hello World!')
    '\n    Taken from zerver.models. Adjusted to work in a migration without changing\n    behavior.\n    '
    try:
        realm_host = host_for_subdomain(realm.string_id)
        validate_email(Address(username='bot', domain=realm_host).addr_spec)
        return realm_host
    except ValidationError:
        pass
    try:
        validate_email(Address(username='bot', domain=settings.FAKE_EMAIL_DOMAIN).addr_spec)
    except ValidationError:
        raise Exception(settings.FAKE_EMAIL_DOMAIN + ' is not a valid domain. Consider setting the FAKE_EMAIL_DOMAIN setting.')
    return settings.FAKE_EMAIL_DOMAIN

def fix_dummy_users(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    '\n    do_delete_users had two bugs:\n    1. Creating the replacement dummy users with active=True\n    2. Creating the replacement dummy users with email domain set to realm.uri,\n    which may not be a valid email domain.\n    Prior commits fixed the bugs, and this migration fixes the pre-existing objects.\n    '
    UserProfile = apps.get_model('zerver', 'UserProfile')
    Subscription = apps.get_model('zerver', 'Subscription')
    users_to_fix = UserProfile.objects.filter(is_mirror_dummy=True, is_active=True, delivery_email__regex='^deleteduser\\d+@.+')
    update_fields = ['is_active']
    for user_profile in users_to_fix:
        user_profile.is_active = False
        try:
            validate_email(user_profile.delivery_email)
        except ValidationError:
            user_profile.delivery_email = Address(username=f'deleteduser{user_profile.id}', domain=get_fake_email_domain(user_profile.realm)).addr_spec
            update_fields.append('delivery_email')
    UserProfile.objects.bulk_update(users_to_fix, update_fields)
    Subscription.objects.filter(user_profile__in=users_to_fix).update(is_user_active=False)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0372_realmemoji_unique_realm_emoji_when_false_deactivated')]
    operations = [migrations.RunPython(fix_dummy_users, reverse_code=migrations.RunPython.noop)]