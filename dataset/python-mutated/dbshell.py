import subprocess
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections

class Command(BaseCommand):
    help = 'Runs the command-line client for specified database, or the default database if none is provided.'
    requires_system_checks = []

    def add_arguments(self, parser):
        if False:
            while True:
                i = 10
        parser.add_argument('--database', default=DEFAULT_DB_ALIAS, help='Nominates a database onto which to open a shell. Defaults to the "default" database.')
        parameters = parser.add_argument_group('parameters', prefix_chars='--')
        parameters.add_argument('parameters', nargs='*')

    def handle(self, **options):
        if False:
            print('Hello World!')
        connection = connections[options['database']]
        try:
            connection.client.runshell(options['parameters'])
        except FileNotFoundError:
            raise CommandError('You appear not to have the %r program installed or on your path.' % connection.client.executable_name)
        except subprocess.CalledProcessError as e:
            raise CommandError('"%s" returned non-zero exit status %s.' % (' '.join(map(str, e.cmd)), e.returncode), returncode=e.returncode)