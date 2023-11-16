import getpass
from argparse import ArgumentParser
from typing import Any, List
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import CommandError
from typing_extensions import override
from zerver.lib.management import ZulipBaseCommand

class Command(ZulipBaseCommand):
    help = "Change a user's password."
    requires_migrations_checks = True
    requires_system_checks: List[str] = []

    def _get_pass(self, prompt: str='Password: ') -> str:
        if False:
            i = 10
            return i + 15
        p = getpass.getpass(prompt=prompt)
        if not p:
            raise CommandError('aborted')
        return p

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        if False:
            for i in range(10):
                print('nop')
        parser.add_argument('email', metavar='<email>', help='email of user to change role')
        self.add_realm_args(parser, required=True)

    @override
    def handle(self, *args: Any, **options: Any) -> str:
        if False:
            return 10
        email = options['email']
        realm = self.get_realm(options)
        u = self.get_user(email, realm)
        self.stdout.write(f"Changing password for user '{u}'")
        MAX_TRIES = 3
        count = 0
        (p1, p2) = ('1', '2')
        password_validated = False
        while (p1 != p2 or not password_validated) and count < MAX_TRIES:
            p1 = self._get_pass()
            p2 = self._get_pass('Password (again): ')
            if p1 != p2:
                self.stdout.write('Passwords do not match. Please try again.')
                count += 1
                continue
            try:
                validate_password(p2, u)
            except ValidationError as err:
                self.stderr.write('\n'.join(err.messages))
                count += 1
            else:
                password_validated = True
        if count == MAX_TRIES:
            raise CommandError(f"Aborting password change for user '{u}' after {count} attempts")
        u.set_password(p1)
        u.save()
        return f"Password changed successfully for user '{u}'"