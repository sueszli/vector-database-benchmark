import json
from pathlib import PosixPath
from django.core.management import BaseCommand
from api_app.analyzers_manager.models import AnalyzerConfig
from api_app.analyzers_manager.serializers import AnalyzerConfigSerializerForMigration
from api_app.connectors_manager.models import ConnectorConfig
from api_app.connectors_manager.serializers import ConnectorConfigSerializerForMigration
from api_app.ingestors_manager.models import IngestorConfig
from api_app.ingestors_manager.serializers import IngestorConfigSerializerForMigration
from api_app.models import PluginConfig
from api_app.pivots_manager.models import PivotConfig
from api_app.pivots_manager.serializers import PivotConfigSerializerForMigration
from api_app.serializers import ParameterCompleteSerializer, PluginConfigCompleteSerializer
from api_app.visualizers_manager.models import VisualizerConfig
from api_app.visualizers_manager.serializers import VisualizerConfigSerializerForMigration

class Command(BaseCommand):
    help = 'Create migration file from plugin saved inside the db'

    @staticmethod
    def add_arguments(parser):
        if False:
            while True:
                i = 10
        parser.add_argument('plugin_class', type=str, help='Plugin config class to use', choices=[AnalyzerConfig.__name__, ConnectorConfig.__name__, VisualizerConfig.__name__, IngestorConfig.__name__, PivotConfig.__name__])
        parser.add_argument('plugin_name', type=str, help='Plugin config name to use')

    @staticmethod
    def _get_serialization(obj, serializer_class):
        if False:
            for i in range(10):
                print('nop')
        obj_data = serializer_class(obj).data
        obj_data['model'] = f'{obj._meta.app_label}.{obj._meta.object_name}'
        params_data = []
        values_data = []
        for parameter in obj.parameters.all():
            params_data.append(ParameterCompleteSerializer(parameter).data)
            try:
                value = PluginConfig.objects.get(owner=None, for_organization=False, parameter=parameter, parameter__is_secret=False)
            except PluginConfig.DoesNotExist:
                ...
            else:
                values_data.append(PluginConfigCompleteSerializer(value).data)
        return (obj_data, params_data, values_data)

    @staticmethod
    def _imports() -> str:
        if False:
            return 10
        return 'from django.db import migrations\nfrom django.db.models.fields.related_descriptors import (\n    ForwardManyToOneDescriptor,\n    ForwardOneToOneDescriptor,\n    ManyToManyDescriptor,\n)\n'

    @staticmethod
    def _migrate_template():
        if False:
            return 10
        return '\ndef _get_real_obj(Model, field, value):\n    if type(getattr(Model, field)) in [ForwardManyToOneDescriptor, ForwardOneToOneDescriptor] and value:\n        other_model = getattr(Model, field).get_queryset().model\n        # in case is a dictionary, we have to retrieve the object with every key\n        if isinstance(value, dict):\n            real_vals = {}\n            for key, real_val in value.items():\n                real_vals[key] = _get_real_obj(other_model, key, real_val)\n            value = other_model.objects.get_or_create(**real_vals)[0]\n        # it is just the primary key serialized\n        else:\n            value = other_model.objects.get(pk=value)\n    return value\n\ndef _create_object(Model, data):\n    mtm, no_mtm = {}, {}\n    for field, value in data.items():\n        if type(getattr(Model, field)) is ManyToManyDescriptor:\n            mtm[field] = value\n        else:\n            value = _get_real_obj(Model, field ,value)\n            no_mtm[field] = value\n    try:\n        o = Model.objects.get(**no_mtm)\n    except Model.DoesNotExist:\n        o = Model(**no_mtm)\n        o.full_clean()\n        o.save()\n        for field, value in mtm.items():\n            attribute = getattr(o, field)\n            attribute.set(value)\n    \ndef migrate(apps, schema_editor):\n    Parameter = apps.get_model("api_app", "Parameter")\n    PluginConfig = apps.get_model("api_app", "PluginConfig")    \n    python_path = plugin.pop("model")\n    Model = apps.get_model(*python_path.split("."))\n    _create_object(Model, plugin)\n    for param in params:\n        _create_object(Parameter, param)\n    for value in values:\n        _create_object(PluginConfig, value)\n\n'

    @staticmethod
    def _reverse_migrate_template():
        if False:
            while True:
                i = 10
        return '\ndef reverse_migrate(apps, schema_editor):\n    python_path = plugin.pop("model")\n    Model = apps.get_model(*python_path.split("."))\n    Model.objects.get(name=plugin["name"]).delete()\n'

    def _body_template(self, app):
        if False:
            i = 10
            return i + 15
        return "\n\nclass Migration(migrations.Migration):\n\n    dependencies = [\n        ('api_app', '{0}'),\n        ('{1}', '{2}'),\n    ]\n\n    operations = [\n        migrations.RunPython(\n            migrate, reverse_migrate\n        )\n    ]\n".format(self._get_last_migration('api_app'), app, self._get_last_migration(app))

    @staticmethod
    def _get_last_migration(app):
        if False:
            for i in range(10):
                print('nop')
        from django.db.migrations.recorder import MigrationRecorder
        return MigrationRecorder.Migration.objects.filter(app=app).latest('id').name

    def _migration_file(self, obj, serializer_class, app):
        if False:
            i = 10
            return i + 15
        (obj_data, param_data, values_data) = self._get_serialization(obj, serializer_class)
        return '{0}\nplugin = {1}\n\nparams = {2}\n\nvalues = {3}\n\n{4}\n{5}\n{6}\n        '.format(self._imports(), str(json.loads(json.dumps(obj_data))), str(json.loads(json.dumps(param_data))), str(json.loads(json.dumps(values_data))), self._migrate_template(), self._reverse_migrate_template(), self._body_template(app))

    def _name_file(self, obj, app):
        if False:
            print('Hello World!')
        from django.db.migrations.autodetector import MigrationAutodetector
        last_migration_number = MigrationAutodetector.parse_number(self._get_last_migration(app))
        return f"{str(int(last_migration_number) + 1).rjust(4, '0')}_{obj.snake_case_name}_{obj.name.lower()}.py"

    @staticmethod
    def _save_file(name_file, content, app):
        if False:
            i = 10
            return i + 15
        path = 'api_app' / PosixPath(app) / 'migrations' / name_file
        if path.exists():
            raise RuntimeError(f'Migration {path} already exists. Please apply migration before create a new one')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def handle(self, *args, **options):
        if False:
            return 10
        config_name = options['plugin_name']
        config_class = options['plugin_class']
        (class_, serializer_class) = (AnalyzerConfig, AnalyzerConfigSerializerForMigration) if config_class == AnalyzerConfig.__name__ else (ConnectorConfig, ConnectorConfigSerializerForMigration) if config_class == ConnectorConfig.__name__ else (VisualizerConfig, VisualizerConfigSerializerForMigration) if config_class == VisualizerConfig.__name__ else (IngestorConfig, IngestorConfigSerializerForMigration) if config_class == IngestorConfig.__name__ else (PivotConfig, PivotConfigSerializerForMigration)
        obj = class_.objects.get(name=config_name)
        app = obj._meta.app_label
        content = self._migration_file(obj, serializer_class, app)
        name_file = self._name_file(obj, app)
        self._save_file(name_file, content, app)
        self.stdout.write(self.style.SUCCESS(f'Migration {name_file} created with success'))