from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from utilities.fields import NaturalOrderingField

class Command(BaseCommand):
    help = 'Recalculate natural ordering values for the specified models'

    def add_arguments(self, parser):
        if False:
            while True:
                i = 10
        parser.add_argument('args', metavar='app_label.ModelName', nargs='*', help='One or more specific models (each prefixed with its app_label) to renaturalize')

    def _get_models(self, names):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compile a list of models to be renaturalized. If no names are specified, all models which have one or more\n        NaturalOrderingFields will be included.\n        '
        models = []
        if names:
            for name in names:
                try:
                    (app_label, model_name) = name.split('.')
                except ValueError:
                    raise CommandError(f'Invalid format: {name}. Models must be specified in the form app_label.ModelName.')
                try:
                    app_config = apps.get_app_config(app_label)
                except LookupError as e:
                    raise CommandError(str(e))
                try:
                    model = app_config.get_model(model_name)
                except LookupError:
                    raise CommandError(f'Unknown model: {app_label}.{model_name}')
                fields = [field for field in model._meta.concrete_fields if type(field) is NaturalOrderingField]
                if not fields:
                    raise CommandError(f'Invalid model: {app_label}.{model_name} does not employ natural ordering')
                models.append((model, fields))
        else:
            for app_config in apps.get_app_configs():
                for model in app_config.models.values():
                    fields = [field for field in model._meta.concrete_fields if type(field) is NaturalOrderingField]
                    if fields:
                        models.append((model, fields))
        return models

    def handle(self, *args, **options):
        if False:
            return 10
        models = self._get_models(args)
        if options['verbosity']:
            self.stdout.write(f'Renaturalizing {len(models)} models.')
        for (model, fields) in models:
            for field in fields:
                target_field = field.target_field
                naturalize = field.naturalize_function
                count = 0
                if options['verbosity']:
                    self.stdout.write(f'{model._meta.label}.{field.target_field} ({field.name})... ', ending='\n' if options['verbosity'] >= 2 else '')
                    self.stdout.flush()
                queryset = model.objects.values_list(target_field, flat=True).order_by(target_field).distinct()
                for value in queryset:
                    naturalized_value = naturalize(value, max_length=field.max_length)
                    if options['verbosity'] >= 2:
                        self.stdout.write(f'  {value} -> {naturalized_value}', ending='')
                        self.stdout.flush()
                    changed = model.objects.filter(name=value).update(**{field.name: naturalized_value})
                    if options['verbosity'] >= 2:
                        self.stdout.write(f' ({changed})')
                    count += changed
                if options['verbosity'] >= 2:
                    self.stdout.write(self.style.SUCCESS(f'{count} {model._meta.verbose_name_plural} updated ({queryset.count()} unique values)'))
                elif options['verbosity']:
                    self.stdout.write(self.style.SUCCESS(str(count)))
        if options['verbosity']:
            self.stdout.write(self.style.SUCCESS('Done.'))