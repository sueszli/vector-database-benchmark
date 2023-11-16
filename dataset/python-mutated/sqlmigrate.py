from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.migrations.loader import AmbiguityError, MigrationLoader

class Command(BaseCommand):
    help = 'Prints the SQL statements for the named migration.'
    output_transaction = True

    def add_arguments(self, parser):
        if False:
            for i in range(10):
                print('nop')
        parser.add_argument('app_label', help='App label of the application containing the migration.')
        parser.add_argument('migration_name', help='Migration name to print the SQL for.')
        parser.add_argument('--database', default=DEFAULT_DB_ALIAS, help='Nominates a database to create SQL for. Defaults to the "default" database.')
        parser.add_argument('--backwards', action='store_true', help='Creates SQL to unapply the migration, rather than to apply it')

    def execute(self, *args, **options):
        if False:
            return 10
        options['no_color'] = True
        return super().execute(*args, **options)

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        connection = connections[options['database']]
        loader = MigrationLoader(connection, replace_migrations=False)
        (app_label, migration_name) = (options['app_label'], options['migration_name'])
        try:
            apps.get_app_config(app_label)
        except LookupError as err:
            raise CommandError(str(err))
        if app_label not in loader.migrated_apps:
            raise CommandError("App '%s' does not have migrations" % app_label)
        try:
            migration = loader.get_migration_by_prefix(app_label, migration_name)
        except AmbiguityError:
            raise CommandError("More than one migration matches '%s' in app '%s'. Please be more specific." % (migration_name, app_label))
        except KeyError:
            raise CommandError("Cannot find a migration matching '%s' from app '%s'. Is it in INSTALLED_APPS?" % (migration_name, app_label))
        target = (app_label, migration.name)
        self.output_transaction = migration.atomic and connection.features.can_rollback_ddl
        plan = [(loader.graph.nodes[target], options['backwards'])]
        sql_statements = loader.collect_sql(plan)
        if not sql_statements and options['verbosity'] >= 1:
            self.stderr.write('No operations found.')
        return '\n'.join(sql_statements)