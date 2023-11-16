from django import forms
from utilities.utils import content_type_name
__all__ = ('ContentTypeChoiceField', 'ContentTypeMultipleChoiceField')

class ContentTypeChoiceMixin:

    def __init__(self, queryset, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        queryset = queryset.order_by('app_label', 'model')
        super().__init__(queryset, *args, **kwargs)

    def label_from_instance(self, obj):
        if False:
            return 10
        try:
            return content_type_name(obj)
        except AttributeError:
            return super().label_from_instance(obj)

class ContentTypeChoiceField(ContentTypeChoiceMixin, forms.ModelChoiceField):
    """
    Selection field for a single content type.
    """
    pass

class ContentTypeMultipleChoiceField(ContentTypeChoiceMixin, forms.ModelMultipleChoiceField):
    """
    Selection field for one or more content types.
    """
    pass