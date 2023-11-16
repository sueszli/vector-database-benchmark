import argparse
from typing import Any
from django.core.exceptions import ValidationError
from django.core.management.base import CommandError
from typing_extensions import override
from zerver.actions.create_realm import do_create_realm
from zerver.actions.create_user import do_create_user
from zerver.forms import check_subdomain_available
from zerver.lib.management import ZulipBaseCommand
from zerver.models import UserProfile

class Command(ZulipBaseCommand):
    help = 'Create a new Zulip organization (realm) via the command line.\n\nWe recommend `./manage.py generate_realm_creation_link` for most\nusers, for several reasons:\n\n* Has a more user-friendly web flow for account creation.\n* Manages passwords in a more natural way.\n* Automatically logs the user in during account creation.\n\nThis management command is available as an alternative for situations\nwhere one wants to script the realm creation process.\n\nSince every Zulip realm must have an owner, this command creates the\ninitial organization owner user for the new realm, using the same\nworkflow as `./manage.py create_user`.\n'

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        if False:
            return 10
        parser.add_argument('realm_name', help='Name for the new organization')
        parser.add_argument('--string-id', help='Subdomain for the new organization. Empty if root domain.', default='')
        parser.add_argument('--allow-reserved-subdomain', action='store_true', help='Allow use of reserved subdomains')
        self.add_create_user_args(parser)

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        realm_name = options['realm_name']
        string_id = options['string_id']
        allow_reserved_subdomain = options['allow_reserved_subdomain']
        try:
            check_subdomain_available(string_id, allow_reserved_subdomain)
        except ValidationError as error:
            raise CommandError(error.message)
        create_user_params = self.get_create_user_params(options)
        try:
            realm = do_create_realm(string_id=string_id, name=realm_name)
        except AssertionError as e:
            raise CommandError(str(e))
        do_create_user(create_user_params.email, create_user_params.password, realm, create_user_params.full_name, role=UserProfile.ROLE_REALM_OWNER, realm_creation=True, tos_version=UserProfile.TOS_VERSION_BEFORE_FIRST_LOGIN, acting_user=None)