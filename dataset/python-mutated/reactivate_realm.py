from argparse import ArgumentParser
from typing import Any
from typing_extensions import override
from zerver.actions.realm_settings import do_reactivate_realm
from zerver.lib.management import ZulipBaseCommand

class Command(ZulipBaseCommand):
    help = 'Script to reactivate a deactivated realm.'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add_realm_args(parser, required=True)

    @override
    def handle(self, *args: Any, **options: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        realm = self.get_realm(options)
        assert realm is not None
        if not realm.deactivated:
            print('Realm', options['realm_id'], 'is already active.')
            return
        print('Reactivating', options['realm_id'])
        do_reactivate_realm(realm)
        print('Done!')