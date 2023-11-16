import logging
from django.db.models.signals import post_save
from django.dispatch import receiver
from netbox.registry import registry
logger = logging.getLogger('netbox.denormalized')

def register(model, field_name, mappings):
    if False:
        for i in range(10):
            print('nop')
    '\n    Register a denormalized model field to ensure that it is kept up-to-date with the related object.\n\n    Args:\n        model: The class being updated\n        field_name: The name of the field related to the triggering instance\n        mappings: Dictionary mapping of local to remote fields\n    '
    logger.debug(f'Registering denormalized field {model}.{field_name}')
    field = model._meta.get_field(field_name)
    rel_model = field.related_model
    registry['denormalized_fields'][rel_model].append((model, field_name, mappings))

@receiver(post_save)
def update_denormalized_fields(sender, instance, created, raw, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Check if the sender has denormalized fields registered, and update them as necessary.\n    '

    def _get_field_value(instance, field_name):
        if False:
            print('Hello World!')
        field = instance._meta.get_field(field_name)
        return field.value_from_object(instance)
    if created or raw:
        return
    for (model, field_name, mappings) in registry['denormalized_fields'].get(sender, []):
        logger.debug(f'Updating denormalized values for {model}.{field_name}')
        filter_params = {field_name: instance.pk}
        update_params = {denorm: _get_field_value(instance, origin) for (denorm, origin) in mappings.items()}
        count = model.objects.filter(**filter_params).update(**update_params)
        logger.debug(f'Updated {count} rows')