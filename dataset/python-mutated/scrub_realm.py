from argparse import ArgumentParser
from typing import Any
from django.core.management.base import CommandError
from typing_extensions import override
from zerver.actions.realm_settings import do_scrub_realm
from zerver.lib.management import ZulipBaseCommand

class Command(ZulipBaseCommand):
    help = 'Script to scrub a deactivated realm.'

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.add_realm_args(parser, required=True)

    @override
    def handle(self, *args: Any, **options: str) -> None:
        if False:
            i = 10
            return i + 15
        realm = self.get_realm(options)
        assert realm is not None
        if not realm.deactivated:
            raise CommandError(f"Realm {options['realm_id']} is active. Please deactivate the realm the first.")
        print('Scrubbing', options['realm_id'])
        do_scrub_realm(realm, acting_user=None)
        print('Done!')