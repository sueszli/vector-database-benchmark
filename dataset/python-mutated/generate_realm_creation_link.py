from typing import Any
from django.core.management.base import CommandError
from django.db import ProgrammingError
from typing_extensions import override
from confirmation.models import generate_realm_creation_url
from zerver.lib.management import ZulipBaseCommand
from zerver.models import Realm

class Command(ZulipBaseCommand):
    help = '\n    Outputs a randomly generated, 1-time-use link for Organization creation.\n    Whoever visits the link can create a new organization on this server, regardless of whether\n    settings.OPEN_REALM_CREATION is enabled. The link would expire automatically after\n    settings.REALM_CREATION_LINK_VALIDITY_DAYS.\n\n    Usage: ./manage.py generate_realm_creation_link '

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            return 10
        try:
            Realm.objects.first()
        except ProgrammingError:
            raise CommandError('The Zulip database does not appear to exist. Have you run initialize-database?')
        url = generate_realm_creation_url(by_admin=True)
        self.stdout.write(self.style.SUCCESS('Please visit the following secure single-use link to register your '))
        self.stdout.write(self.style.SUCCESS('new Zulip organization:\x1b[0m'))
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS(f'    \x1b[1;92m{url}\x1b[0m'))
        self.stdout.write('')