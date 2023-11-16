from django.core.management.base import BaseCommand, CommandError
from core.models import DataSource

class Command(BaseCommand):
    help = 'Synchronize a data source from its remote upstream'

    def add_arguments(self, parser):
        if False:
            return 10
        parser.add_argument('name', nargs='*', help='Data source(s) to synchronize')
        parser.add_argument('--all', action='store_true', dest='sync_all', help='Synchronize all data sources')

    def handle(self, *args, **options):
        if False:
            return 10
        if options['sync_all']:
            datasources = DataSource.objects.all()
        elif options['name']:
            datasources = DataSource.objects.filter(name__in=options['name'])
            found_names = {ds['name'] for ds in datasources.values('name')}
            if (invalid_names := (set(options['name']) - found_names)):
                raise CommandError(f"Invalid data source names: {', '.join(invalid_names)}")
        else:
            raise CommandError(f'Must specify at least one data source, or set --all.')
        if len(options['name']) > 1:
            self.stdout.write(f'Syncing {len(datasources)} data sources.')
        for (i, datasource) in enumerate(datasources, start=1):
            self.stdout.write(f'[{i}] Syncing {datasource}... ', ending='')
            self.stdout.flush()
            datasource.sync()
            self.stdout.write(datasource.get_status_display())
            self.stdout.flush()
        if len(options['name']) > 1:
            self.stdout.write(f'Finished.')