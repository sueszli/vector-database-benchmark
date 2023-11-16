from awx.api import serializers
from awx.main.models import UnifiedJob, UnifiedJobTemplate
from rest_framework.generics import ListAPIView

def test_unified_template_field_consistency():
    if False:
        i = 10
        return i + 15
    '\n    Example of what is being tested:\n    The endpoints /projects/N/ and /projects/ should have the same fields as\n    that same project when it is serialized by the unified job template serializer\n    in /unified_job_templates/\n    '
    for cls in UnifiedJobTemplate.__subclasses__():
        detail_serializer = getattr(serializers, '{}Serializer'.format(cls.__name__))
        unified_serializer = serializers.UnifiedJobTemplateSerializer().get_sub_serializer(cls())
        assert set(detail_serializer().fields.keys()) == set(unified_serializer().fields.keys())

def test_unified_job_list_field_consistency():
    if False:
        for i in range(10):
            print('nop')
    '\n    Example of what is being tested:\n    The endpoint /project_updates/ should have the same fields as that\n    project update when it is serialized by the unified job template serializer\n    in /unified_jobs/\n    '
    for cls in UnifiedJob.__subclasses__():
        list_serializer = getattr(serializers, '{}ListSerializer'.format(cls.__name__))
        unified_serializer = serializers.UnifiedJobListSerializer().get_sub_serializer(cls())
        assert set(list_serializer().fields.keys()) == set(unified_serializer().fields.keys()), 'Mismatch between {} list serializer & unified list serializer'.format(cls)

def test_unified_job_detail_exclusive_fields():
    if False:
        print('Hello World!')
    '\n    For each type, assert that the only fields allowed to be exclusive to\n    detail view are the allowed types\n    '
    allowed_detail_fields = frozenset(('result_traceback', 'job_args', 'job_cwd', 'job_env', 'event_processing_finished'))
    for cls in UnifiedJob.__subclasses__():
        list_serializer = getattr(serializers, '{}ListSerializer'.format(cls.__name__))
        detail_serializer = getattr(serializers, '{}Serializer'.format(cls.__name__))
        list_fields = set(list_serializer().fields.keys())
        detail_fields = set(detail_serializer().fields.keys()) - allowed_detail_fields
        assert list_fields == detail_fields, 'List / detail mismatch for serializers of {}'.format(cls)

def test_list_views_use_list_serializers(all_views):
    if False:
        while True:
            i = 10
    '\n    Check that the list serializers are only used for list views,\n    and vice versa\n    '
    list_serializers = tuple((getattr(serializers, '{}ListSerializer'.format(cls.__name__)) for cls in UnifiedJob.__subclasses__() + [UnifiedJob]))
    for View in all_views:
        if hasattr(View, 'model') and type(View.model) is not property and issubclass(getattr(View, 'model'), UnifiedJob):
            if issubclass(View, ListAPIView):
                assert issubclass(View.serializer_class, list_serializers), 'View {} serializer {} is not a list serializer'.format(View, View.serializer_class)
            else:
                assert not issubclass(View.model, list_serializers)