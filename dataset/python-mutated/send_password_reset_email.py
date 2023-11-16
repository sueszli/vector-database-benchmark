from argparse import ArgumentParser
from typing import Any, Iterable
from django.contrib.auth.tokens import default_token_generator
from django.core.management.base import CommandError
from typing_extensions import override
from zerver.forms import generate_password_reset_url
from zerver.lib.management import ZulipBaseCommand
from zerver.lib.send_email import FromAddress, send_email
from zerver.models import UserProfile

class Command(ZulipBaseCommand):
    help = 'Send email to specified email address.'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            print('Hello World!')
        parser.add_argument('--entire-server', action='store_true', help='Send to every user on the server. ')
        self.add_user_list_args(parser, help='Email addresses of user(s) to send password reset emails to.', all_users_help='Send to every user on the realm.')
        self.add_realm_args(parser)

    @override
    def handle(self, *args: Any, **options: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if options['entire_server']:
            users: Iterable[UserProfile] = UserProfile.objects.filter(is_active=True, is_bot=False, is_mirror_dummy=False)
        else:
            realm = self.get_realm(options)
            try:
                users = self.get_users(options, realm, is_bot=False)
            except CommandError as error:
                if str(error) == 'You have to pass either -u/--users or -a/--all-users.':
                    raise CommandError('You have to pass -u/--users or -a/--all-users or --entire-server.')
                raise error
        self.send(users)

    def send(self, users: Iterable[UserProfile]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sends one-use only links for resetting password to target users'
        for user_profile in users:
            context = {'email': user_profile.delivery_email, 'reset_url': generate_password_reset_url(user_profile, default_token_generator), 'realm_uri': user_profile.realm.uri, 'realm_name': user_profile.realm.name, 'active_account_in_realm': True}
            send_email('zerver/emails/password_reset', to_user_ids=[user_profile.id], from_address=FromAddress.tokenized_no_reply_address(), from_name=FromAddress.security_email_from_name(user_profile=user_profile), context=context)