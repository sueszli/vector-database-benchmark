from django.core.management import BaseCommand
from api_app.analyzers_manager.models import AnalyzerConfig

class Command(BaseCommand):
    help = 'Execute celery update task'

    @staticmethod
    def add_arguments(parser):
        if False:
            i = 10
            return i + 15
        parser.add_argument('config_names', nargs='+', help='Analyzers config name to use')

    def handle(self, *args, **options):
        if False:
            i = 10
            return i + 15
        for analyzer_config in AnalyzerConfig.objects.filter(name=options['config_name']).annotate_runnable():
            if analyzer_config.runnable:
                class_ = analyzer_config.python_class
                self.stdout.write(self.style.SUCCESS(f'Starting update of {analyzer_config.name}'))
                if class_.update():
                    self.stdout.write(self.style.SUCCESS(f'Finished update of {analyzer_config.name}'))
                else:
                    self.stdout.write(self.style.WARNING(f'Configuration {analyzer_config.name} does not implement _update method'))
            else:
                self.stdout.write(self.style.WARNING(f'Configuration {analyzer_config.name} is not runnable'))
        self.stdout.write(self.style.SUCCESS('Finish execution'))