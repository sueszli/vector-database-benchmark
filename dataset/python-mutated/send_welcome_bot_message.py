from argparse import ArgumentParser
from typing import Any
from typing_extensions import override
from zerver.lib.management import ZulipBaseCommand
from zerver.lib.onboarding import send_initial_direct_message

class Command(ZulipBaseCommand):
    help = 'Sends the initial welcome bot message.'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            while True:
                i = 10
        self.add_user_list_args(parser, help='Email addresses of user(s) to send welcome bot messages to.', all_users_help='Send to every user on the realm.')
        self.add_realm_args(parser)

    @override
    def handle(self, *args: Any, **options: str) -> None:
        if False:
            while True:
                i = 10
        for user_profile in self.get_users(options, self.get_realm(options), is_bot=False):
            send_initial_direct_message(user_profile)