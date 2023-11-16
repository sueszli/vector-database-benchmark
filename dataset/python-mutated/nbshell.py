import code
import platform
import sys
from django import get_version
from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand
APPS = ('circuits', 'core', 'dcim', 'extras', 'ipam', 'tenancy', 'users', 'virtualization', 'wireless')
BANNER_TEXT = '### NetBox interactive shell ({node})\n### Python {python} | Django {django} | NetBox {netbox}\n### lsmodels() will show available models. Use help(<model>) for more info.'.format(node=platform.node(), python=platform.python_version(), django=get_version(), netbox=settings.VERSION)

class Command(BaseCommand):
    help = 'Start the Django shell with all NetBox models already imported'
    django_models = {}

    def add_arguments(self, parser):
        if False:
            print('Hello World!')
        parser.add_argument('-c', '--command', help='Python code to execute (instead of starting an interactive shell)')

    def _lsmodels(self):
        if False:
            i = 10
            return i + 15
        for (app, models) in self.django_models.items():
            app_name = apps.get_app_config(app).verbose_name
            print(f'{app_name}:')
            for m in models:
                print(f'  {m}')

    def get_namespace(self):
        if False:
            i = 10
            return i + 15
        namespace = {}
        for app in APPS:
            self.django_models[app] = []
            for model in apps.get_app_config(app).get_models():
                namespace[model.__name__] = model
                self.django_models[app].append(model.__name__)
            try:
                app_constants = sys.modules[f'{app}.constants']
                for name in dir(app_constants):
                    namespace[name] = getattr(app_constants, name)
            except KeyError:
                pass
        namespace['ContentType'] = ContentType
        namespace['User'] = get_user_model()
        namespace.update({'lsmodels': self._lsmodels})
        return namespace

    def handle(self, **options):
        if False:
            for i in range(10):
                print('nop')
        namespace = self.get_namespace()
        if options['command']:
            exec(options['command'], namespace)
            return
        try:
            import readline
            import rlcompleter
        except ModuleNotFoundError:
            pass
        else:
            readline.set_completer(rlcompleter.Completer(namespace).complete)
            readline.parse_and_bind('tab: complete')
        shell = code.interact(banner=BANNER_TEXT, local=namespace)
        return shell