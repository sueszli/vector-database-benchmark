from django.contrib.contenttypes.models import ContentType
from wagtail.coreutils import resolve_model_string
CONTENT_TYPE_ORDER = {}

def register(model, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Registers order against the model content_type, used to\n    control the order the models and its permissions appear\n    in the groups object permission editor\n    '
    order = kwargs.pop('order', None)
    if order is not None:
        content_type = ContentType.objects.get_for_model(resolve_model_string(model))
        CONTENT_TYPE_ORDER[content_type.id] = order