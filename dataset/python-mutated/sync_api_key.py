import os
from configparser import ConfigParser
from typing import Any
from django.core.management.base import BaseCommand
from typing_extensions import override
from zerver.models import UserProfile, get_realm, get_user_by_delivery_email

class Command(BaseCommand):
    help = 'Sync your API key from ~/.zuliprc into your development instance'

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            print('Hello World!')
        config_file = os.path.join(os.environ['HOME'], '.zuliprc')
        if not os.path.exists(config_file):
            raise RuntimeError('No ~/.zuliprc found')
        config = ConfigParser()
        with open(config_file) as f:
            config.read_file(f, config_file)
        api_key = config.get('api', 'key')
        email = config.get('api', 'email')
        try:
            realm = get_realm('zulip')
            user_profile = get_user_by_delivery_email(email, realm)
            user_profile.api_key = api_key
            user_profile.save(update_fields=['api_key'])
        except UserProfile.DoesNotExist:
            print(f'User {email} does not exist; not syncing API key')