from django.contrib.contenttypes.models import ContentType
from django.db import models
from awx.main.utils.common import camelcase_to_underscore

def build_polymorphic_ctypes_map(cls):
    if False:
        i = 10
        return i + 15
    mapping = {}
    for ct in ContentType.objects.filter(app_label='main'):
        ct_model_class = ct.model_class()
        if ct_model_class and issubclass(ct_model_class, cls):
            mapping[ct.id] = camelcase_to_underscore(ct_model_class.__name__)
    return mapping

def SET_NULL(collector, field, sub_objs, using):
    if False:
        i = 10
        return i + 15
    if hasattr(sub_objs, 'non_polymorphic'):
        sub_objs = sub_objs.non_polymorphic()
    return models.SET_NULL(collector, field, sub_objs, using)