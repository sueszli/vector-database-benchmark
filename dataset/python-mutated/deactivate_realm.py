from argparse import ArgumentParser
from typing import Any
from typing_extensions import override
from zerver.actions.realm_settings import do_add_deactivated_redirect, do_deactivate_realm
from zerver.lib.management import ZulipBaseCommand

class Command(ZulipBaseCommand):
    help = 'Script to deactivate a realm.'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            print('Hello World!')
        parser.add_argument('--redirect_url', metavar='<redirect_url>', help='URL to which the realm has moved')
        self.add_realm_args(parser, required=True)

    @override
    def handle(self, *args: Any, **options: str) -> None:
        if False:
            i = 10
            return i + 15
        realm = self.get_realm(options)
        assert realm is not None
        if options['redirect_url']:
            print('Setting the redirect URL to', options['redirect_url'])
            do_add_deactivated_redirect(realm, options['redirect_url'])
        if realm.deactivated:
            print('The realm', options['realm_id'], 'is already deactivated.')
            return
        print('Deactivating', options['realm_id'])
        do_deactivate_realm(realm, acting_user=None)
        print('Done!')