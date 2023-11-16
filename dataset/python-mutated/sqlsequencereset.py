from django.core.management.base import AppCommand
from django.db import DEFAULT_DB_ALIAS, connections

class Command(AppCommand):
    help = 'Prints the SQL statements for resetting sequences for the given app name(s).'
    output_transaction = True

    def add_arguments(self, parser):
        if False:
            return 10
        super().add_arguments(parser)
        parser.add_argument('--database', default=DEFAULT_DB_ALIAS, help='Nominates a database to print the SQL for. Defaults to the "default" database.')

    def handle_app_config(self, app_config, **options):
        if False:
            for i in range(10):
                print('nop')
        if app_config.models_module is None:
            return
        connection = connections[options['database']]
        models = app_config.get_models(include_auto_created=True)
        statements = connection.ops.sequence_reset_sql(self.style, models)
        if not statements and options['verbosity'] >= 1:
            self.stderr.write('No sequences found.')
        return '\n'.join(statements)