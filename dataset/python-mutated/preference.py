from rest_framework import generics
from rest_framework.serializers import Serializer
from common.permissions import IsValidUser
from common.utils import get_logger
from .. import serializers
from ..models import Preference
logger = get_logger(__file__)

class PreferenceApi(generics.RetrieveUpdateAPIView):
    permission_classes = (IsValidUser,)
    queryset = Preference.objects.all()
    serializer_class_mapper = {'lina': serializers.LinaSerializer, 'luna': serializers.LunaSerializer, 'koko': serializers.KokoSerializer}

    def check_permissions(self, request):
        if False:
            return 10
        if self.category not in self.serializer_class_mapper:
            return self.permission_denied(request, 'category is invalid')
        return super().check_permissions(request)

    @property
    def user(self):
        if False:
            for i in range(10):
                print('nop')
        return self.request.user

    @property
    def category(self):
        if False:
            print('Hello World!')
        return self.request.query_params.get('category')

    def get_serializer_class(self):
        if False:
            while True:
                i = 10
        cls = self.serializer_class_mapper.get(self.category)
        return cls

    def get_field_defaults(self, serializer):
        if False:
            return 10
        field_defaults = {}
        fields = serializer.get_fields()
        for (name, field) in fields.items():
            if isinstance(field, Serializer):
                field_defaults[name] = self.get_field_defaults(field)
                continue
            field_defaults[name] = getattr(field, 'default', None)
        return field_defaults

    def get_encrypted_fields(self, serializer):
        if False:
            print('Hello World!')
        encrypted_fields = []
        fields = serializer.get_fields()
        for (name, field) in fields.items():
            if isinstance(field, Serializer):
                encrypted_fields += self.get_encrypted_fields(field)
                continue
            if not field.write_only:
                continue
            encrypted_fields.append(name)
        return encrypted_fields

    def get_object(self):
        if False:
            while True:
                i = 10
        serializer = self.get_serializer_class()()
        field_defaults = self.get_field_defaults(serializer)
        qs = self.queryset.filter(user=self.user, category=self.category)
        if not qs.exists():
            return field_defaults
        data = dict(qs.values_list('name', 'value'))
        for (k, v) in data.items():
            for d in field_defaults.values():
                if k in d:
                    d[k] = v
                    break
        return field_defaults

    def perform_update(self, serializer):
        if False:
            return 10
        user = self.user
        category = self.category
        model = self.queryset.model
        encrypted_fields = self.get_encrypted_fields(serializer)
        data = serializer.validated_data
        for d in data.values():
            for (name, value) in d.items():
                kwargs = {'name': name, 'user': user}
                defaults = {'category': category}
                if name in encrypted_fields:
                    value = model.encrypt(value)
                    defaults['encrypted'] = True
                defaults['value'] = value
                defaults.update(kwargs)
                model.objects.update_or_create(defaults, **kwargs)