from collections import defaultdict
from django.core.management.base import BaseCommand
from django.db.models import Count, OuterRef, Subquery
from netbox.registry import registry
from utilities.counters import update_counts

class Command(BaseCommand):
    help = 'Force a recalculation of all cached counter fields'

    @staticmethod
    def collect_models():
        if False:
            for i in range(10):
                print('nop')
        '\n        Query the registry to find all models which have one or more counter fields. Return a mapping of counter fields\n        to related query names for each model.\n        '
        models = defaultdict(dict)
        for (model, field_mappings) in registry['counter_fields'].items():
            for (field_name, counter_name) in field_mappings.items():
                fk_field = model._meta.get_field(field_name)
                parent_model = fk_field.related_model
                related_query_name = fk_field.related_query_name()
                models[parent_model][counter_name] = related_query_name
        return models

    def handle(self, *model_names, **options):
        if False:
            return 10
        for (model, mappings) in self.collect_models().items():
            for (field_name, related_query) in mappings.items():
                update_counts(model, field_name, related_query)
        self.stdout.write(self.style.SUCCESS('Finished.'))