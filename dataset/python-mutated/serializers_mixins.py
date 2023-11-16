class HideSensitiveFieldsSerializerMixin:

    def to_representation(self, instance):
        if False:
            i = 10
            return i + 15
        data = super().to_representation(instance)
        environment = self.context['request'].environment
        if environment.hide_sensitive_data:
            for field in self.sensitive_fields:
                data[field] = [] if isinstance(data[field], list) else None
        return data