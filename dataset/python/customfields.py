from django.contrib.contenttypes.models import ContentType
from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes
from rest_framework.fields import Field
from rest_framework.serializers import ValidationError

from extras.choices import CustomFieldTypeChoices
from extras.models import CustomField
from netbox.constants import NESTED_SERIALIZER_PREFIX
from utilities.api import get_serializer_for_model


#
# Custom fields
#

class CustomFieldDefaultValues:
    """
    Return a dictionary of all CustomFields assigned to the parent model and their default values.
    """
    requires_context = True

    def __call__(self, serializer_field):
        self.model = serializer_field.parent.Meta.model

        # Retrieve the CustomFields for the parent model
        content_type = ContentType.objects.get_for_model(self.model)
        fields = CustomField.objects.filter(content_types=content_type)

        # Populate the default value for each CustomField
        value = {}
        for field in fields:
            if field.default is not None:
                value[field.name] = field.default
            else:
                value[field.name] = None

        return value


@extend_schema_field(OpenApiTypes.OBJECT)
class CustomFieldsDataField(Field):

    def _get_custom_fields(self):
        """
        Cache CustomFields assigned to this model to avoid redundant database queries
        """
        if not hasattr(self, '_custom_fields'):
            content_type = ContentType.objects.get_for_model(self.parent.Meta.model)
            self._custom_fields = CustomField.objects.filter(content_types=content_type)
        return self._custom_fields

    def to_representation(self, obj):
        # TODO: Fix circular import
        from utilities.api import get_serializer_for_model
        data = {}
        for cf in self._get_custom_fields():
            value = cf.deserialize(obj.get(cf.name))
            if value is not None and cf.type == CustomFieldTypeChoices.TYPE_OBJECT:
                serializer = get_serializer_for_model(cf.object_type.model_class(), prefix=NESTED_SERIALIZER_PREFIX)
                value = serializer(value, context=self.parent.context).data
            elif value is not None and cf.type == CustomFieldTypeChoices.TYPE_MULTIOBJECT:
                serializer = get_serializer_for_model(cf.object_type.model_class(), prefix=NESTED_SERIALIZER_PREFIX)
                value = serializer(value, many=True, context=self.parent.context).data
            data[cf.name] = value

        return data

    def to_internal_value(self, data):
        if type(data) is not dict:
            raise ValidationError(
                "Invalid data format. Custom field data must be passed as a dictionary mapping field names to their "
                "values."
            )

        # Serialize object and multi-object values
        for cf in self._get_custom_fields():
            if cf.name in data and data[cf.name] not in (None, []) and cf.type in (
                    CustomFieldTypeChoices.TYPE_OBJECT,
                    CustomFieldTypeChoices.TYPE_MULTIOBJECT
            ):
                serializer_class = get_serializer_for_model(
                    model=cf.object_type.model_class(),
                    prefix=NESTED_SERIALIZER_PREFIX
                )
                many = cf.type == CustomFieldTypeChoices.TYPE_MULTIOBJECT
                serializer = serializer_class(data=data[cf.name], many=many, context=self.parent.context)
                if serializer.is_valid():
                    data[cf.name] = [obj['id'] for obj in serializer.data] if many else serializer.data['id']
                else:
                    raise ValidationError(f"Unknown related object(s): {data[cf.name]}")

        # If updating an existing instance, start with existing custom_field_data
        if self.parent.instance:
            data = {**self.parent.instance.custom_field_data, **data}

        return data
