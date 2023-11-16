from django.contrib.contenttypes.models import ContentType
from django.db.models import DEFERRED
from django.utils.functional import cached_property

class SpecificMixin:
    """
    Mixin for models that support multi-table inheritance and provide a
    ``content_type`` field pointing to the specific model class, to provide
    methods and properties for retrieving the specific instance of the model.
    """

    def get_specific(self, deferred=False, copy_attrs=None, copy_attrs_exclude=None):
        if False:
            i = 10
            return i + 15
        '\n        Return this object in its most specific subclassed form.\n\n        By default, a database query is made to fetch all field values for the\n        specific object. If you only require access to custom methods or other\n        non-field attributes on the specific object, you can use\n        ``deferred=True`` to avoid this query. However, any attempts to access\n        specific field values from the returned object will trigger additional\n        database queries.\n\n        By default, references to all non-field attribute values are copied\n        from current object to the returned one. This includes:\n\n        * Values set by a queryset, for example: annotations, or values set as\n          a result of using ``select_related()`` or ``prefetch_related()``.\n        * Any ``cached_property`` values that have been evaluated.\n        * Attributes set elsewhere in Python code.\n\n        For fine-grained control over which non-field values are copied to the\n        returned object, you can use ``copy_attrs`` to specify a complete list\n        of attribute names to include. Alternatively, you can use\n        ``copy_attrs_exclude`` to specify a list of attribute names to exclude.\n\n        If called on an object that is already an instance of the most specific\n        class, the object will be returned as is, and no database queries or\n        other operations will be triggered.\n\n        If the object was originally created using a model that has since\n        been removed from the codebase, an instance of the base class will be\n        returned (without any custom field values or other functionality\n        present on the original class). Usually, deleting these objects is the\n        best course of action, but there is currently no safe way for Wagtail\n        to do that at migration time.\n        '
        model_class = self.specific_class
        if model_class is None:
            return self
        if isinstance(self, model_class):
            return self
        if deferred:
            values = tuple((getattr(self, f.attname, self.pk if f.primary_key else DEFERRED) for f in model_class._meta.concrete_fields))
            specific_obj = model_class(*values)
            specific_obj._state.adding = self._state.adding
        else:
            specific_obj = model_class._default_manager.get(id=self.id)
        if copy_attrs is not None:
            for attr in (attr for attr in copy_attrs if attr in self.__dict__):
                setattr(specific_obj, attr, getattr(self, attr))
        else:
            exclude = copy_attrs_exclude or ()
            for (k, v) in ((k, v) for (k, v) in self.__dict__.items() if k not in exclude):
                specific_obj.__dict__.setdefault(k, v)
        return specific_obj

    @cached_property
    def specific(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns this object in its most specific subclassed form with all field\n        values fetched from the database. The result is cached in memory.\n        '
        return self.get_specific()

    @cached_property
    def specific_deferred(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns this object in its most specific subclassed form without any\n        additional field values being fetched from the database. The result\n        is cached in memory.\n        '
        return self.get_specific(deferred=True)

    @cached_property
    def specific_class(self):
        if False:
            while True:
                i = 10
        '\n        Return the class that this object would be if instantiated in its\n        most specific form.\n\n        If the model class can no longer be found in the codebase, and the\n        relevant ``ContentType`` has been removed by a database migration,\n        the return value will be ``None``.\n\n        If the model class can no longer be found in the codebase, but the\n        relevant ``ContentType`` is still present in the database (usually a\n        result of switching between git branches without running or reverting\n        database migrations beforehand), the return value will be ``None``.\n        '
        return self.cached_content_type.model_class()

    @property
    def cached_content_type(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return this object's ``content_type`` value from the ``ContentType``\n        model's cached manager, which will avoid a database query if the\n        content type is already in memory.\n        "
        return ContentType.objects.get_for_id(self.content_type_id)