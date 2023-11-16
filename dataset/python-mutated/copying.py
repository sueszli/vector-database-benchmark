from django.contrib.contenttypes.fields import GenericRelation
from django.db import models
from modelcluster.fields import ParentalKey, ParentalManyToManyField
from modelcluster.models import ClusterableModel

def _extract_field_data(source, exclude_fields=None):
    if False:
        i = 10
        return i + 15
    "\n    Get dictionaries representing the model's field data.\n\n    This excludes many to many fields (which are handled by _copy_m2m_relations)'\n    "
    exclude_fields = exclude_fields or []
    data_dict = {}
    for field in source._meta.get_fields():
        if field.name in exclude_fields:
            continue
        if field.auto_created:
            continue
        if isinstance(field, GenericRelation):
            continue
        if field.many_to_many:
            if isinstance(field, ParentalManyToManyField):
                parental_field = getattr(source, field.name)
                if hasattr(parental_field, 'all'):
                    values = parental_field.all()
                    if values:
                        data_dict[field.name] = values
            continue
        if isinstance(field, models.OneToOneField) and field.remote_field.parent_link:
            continue
        if isinstance(field, models.ForeignKey):
            data_dict[field.name] = None
            data_dict[field.attname] = getattr(source, field.attname)
        else:
            data_dict[field.name] = getattr(source, field.name)
    return data_dict

def _copy_m2m_relations(source, target, exclude_fields=None, update_attrs=None):
    if False:
        while True:
            i = 10
    '\n    Copies non-ParentalManyToMany m2m relations\n    '
    update_attrs = update_attrs or {}
    exclude_fields = exclude_fields or []
    for field in source._meta.get_fields():
        if field.many_to_many and field.name not in exclude_fields and (not field.auto_created) and (not isinstance(field, ParentalManyToManyField)):
            try:
                through_model_parental_links = [field for field in field.through._meta.get_fields() if isinstance(field, ParentalKey) and issubclass(source.__class__, field.related_model)]
                if through_model_parental_links:
                    continue
            except AttributeError:
                pass
            if field.name in update_attrs:
                value = update_attrs[field.name]
            else:
                value = getattr(source, field.name).all()
            getattr(target, field.name).set(value)

def _copy(source, exclude_fields=None, update_attrs=None):
    if False:
        i = 10
        return i + 15
    data_dict = _extract_field_data(source, exclude_fields=exclude_fields)
    target = source.__class__(**data_dict)
    if update_attrs:
        for (field, value) in update_attrs.items():
            if field not in data_dict:
                continue
            setattr(target, field, value)
    if isinstance(source, ClusterableModel):
        child_object_map = source.copy_all_child_relations(target, exclude=exclude_fields)
    else:
        child_object_map = {}
    return (target, child_object_map)