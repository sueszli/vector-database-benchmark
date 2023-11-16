"""
"Rel objects" for related fields.

"Rel objects" (for lack of a better name) carry information about the relation
modeled by a related field and provide some utility functions. They're stored
in the ``remote_field`` attribute of the field.

They also act as reverse fields for the purposes of the Meta API because
they're the closest concept currently available.
"""
import warnings
from django.core import exceptions
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from . import BLANK_CHOICE_DASH
from .mixins import FieldCacheMixin

class ForeignObjectRel(FieldCacheMixin):
    """
    Used by ForeignObject to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """
    auto_created = True
    concrete = False
    editable = False
    is_relation = True
    null = True
    empty_strings_allowed = False

    def __init__(self, field, to, related_name=None, related_query_name=None, limit_choices_to=None, parent_link=False, on_delete=None):
        if False:
            for i in range(10):
                print('nop')
        self.field = field
        self.model = to
        self.related_name = related_name
        self.related_query_name = related_query_name
        self.limit_choices_to = {} if limit_choices_to is None else limit_choices_to
        self.parent_link = parent_link
        self.on_delete = on_delete
        self.symmetrical = False
        self.multiple = True

    @cached_property
    def hidden(self):
        if False:
            return 10
        return self.is_hidden()

    @cached_property
    def name(self):
        if False:
            return 10
        return self.field.related_query_name()

    @property
    def remote_field(self):
        if False:
            while True:
                i = 10
        return self.field

    @property
    def target_field(self):
        if False:
            return 10
        '\n        When filtering against this relation, return the field on the remote\n        model against which the filtering should happen.\n        '
        target_fields = self.path_infos[-1].target_fields
        if len(target_fields) > 1:
            raise exceptions.FieldError("Can't use target_field for multicolumn relations.")
        return target_fields[0]

    @cached_property
    def related_model(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.field.model:
            raise AttributeError("This property can't be accessed before self.field.contribute_to_class has been called.")
        return self.field.model

    @cached_property
    def many_to_many(self):
        if False:
            while True:
                i = 10
        return self.field.many_to_many

    @cached_property
    def many_to_one(self):
        if False:
            print('Hello World!')
        return self.field.one_to_many

    @cached_property
    def one_to_many(self):
        if False:
            i = 10
            return i + 15
        return self.field.many_to_one

    @cached_property
    def one_to_one(self):
        if False:
            i = 10
            return i + 15
        return self.field.one_to_one

    def get_lookup(self, lookup_name):
        if False:
            for i in range(10):
                print('nop')
        return self.field.get_lookup(lookup_name)

    def get_lookups(self):
        if False:
            i = 10
            return i + 15
        return self.field.get_lookups()

    def get_transform(self, name):
        if False:
            print('Hello World!')
        return self.field.get_transform(name)

    def get_internal_type(self):
        if False:
            i = 10
            return i + 15
        return self.field.get_internal_type()

    @property
    def db_type(self):
        if False:
            print('Hello World!')
        return self.field.db_type

    def __repr__(self):
        if False:
            return 10
        return '<%s: %s.%s>' % (type(self).__name__, self.related_model._meta.app_label, self.related_model._meta.model_name)

    @property
    def identity(self):
        if False:
            i = 10
            return i + 15
        return (self.field, self.model, self.related_name, self.related_query_name, make_hashable(self.limit_choices_to), self.parent_link, self.on_delete, self.symmetrical, self.multiple)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.identity)

    def __getstate__(self):
        if False:
            print('Hello World!')
        state = self.__dict__.copy()
        state.pop('path_infos', None)
        return state

    def get_choices(self, include_blank=True, blank_choice=BLANK_CHOICE_DASH, limit_choices_to=None, ordering=()):
        if False:
            print('Hello World!')
        '\n        Return choices with a default blank choices included, for use\n        as <select> choices for this field.\n\n        Analog of django.db.models.fields.Field.get_choices(), provided\n        initially for utilization by RelatedFieldListFilter.\n        '
        limit_choices_to = limit_choices_to or self.limit_choices_to
        qs = self.related_model._default_manager.complex_filter(limit_choices_to)
        if ordering:
            qs = qs.order_by(*ordering)
        return (blank_choice if include_blank else []) + [(x.pk, str(x)) for x in qs]

    def is_hidden(self):
        if False:
            print('Hello World!')
        'Should the related object be hidden?'
        return bool(self.related_name) and self.related_name[-1] == '+'

    def get_joining_columns(self):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('ForeignObjectRel.get_joining_columns() is deprecated. Use get_joining_fields() instead.', RemovedInDjango60Warning)
        return self.field.get_reverse_joining_columns()

    def get_joining_fields(self):
        if False:
            print('Hello World!')
        return self.field.get_reverse_joining_fields()

    def get_extra_restriction(self, alias, related_alias):
        if False:
            i = 10
            return i + 15
        return self.field.get_extra_restriction(related_alias, alias)

    def set_field_name(self):
        if False:
            while True:
                i = 10
        "\n        Set the related field's name, this is not available until later stages\n        of app loading, so set_field_name is called from\n        set_attributes_from_rel()\n        "
        self.field_name = None

    def get_accessor_name(self, model=None):
        if False:
            i = 10
            return i + 15
        opts = model._meta if model else self.related_model._meta
        model = model or self.related_model
        if self.multiple:
            if self.symmetrical and model == self.model:
                return None
        if self.related_name:
            return self.related_name
        return opts.model_name + ('_set' if self.multiple else '')

    def get_path_info(self, filtered_relation=None):
        if False:
            print('Hello World!')
        if filtered_relation:
            return self.field.get_reverse_path_info(filtered_relation)
        else:
            return self.field.reverse_path_infos

    @cached_property
    def path_infos(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_path_info()

    def get_cache_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the name of the cache key to use for storing an instance of the\n        forward model on the reverse model.\n        '
        return self.get_accessor_name()

class ManyToOneRel(ForeignObjectRel):
    """
    Used by the ForeignKey field to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.

    Note: Because we somewhat abuse the Rel objects by using them as reverse
    fields we get the funny situation where
    ``ManyToOneRel.many_to_one == False`` and
    ``ManyToOneRel.one_to_many == True``. This is unfortunate but the actual
    ManyToOneRel class is a private API and there is work underway to turn
    reverse relations into actual fields.
    """

    def __init__(self, field, to, field_name, related_name=None, related_query_name=None, limit_choices_to=None, parent_link=False, on_delete=None):
        if False:
            while True:
                i = 10
        super().__init__(field, to, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, parent_link=parent_link, on_delete=on_delete)
        self.field_name = field_name

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = super().__getstate__()
        state.pop('related_model', None)
        return state

    @property
    def identity(self):
        if False:
            return 10
        return super().identity + (self.field_name,)

    def get_related_field(self):
        if False:
            return 10
        "\n        Return the Field in the 'to' object to which this relationship is tied.\n        "
        field = self.model._meta.get_field(self.field_name)
        if not field.concrete:
            raise exceptions.FieldDoesNotExist("No related field named '%s'" % self.field_name)
        return field

    def set_field_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.field_name = self.field_name or self.model._meta.pk.name

class OneToOneRel(ManyToOneRel):
    """
    Used by OneToOneField to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    def __init__(self, field, to, field_name, related_name=None, related_query_name=None, limit_choices_to=None, parent_link=False, on_delete=None):
        if False:
            while True:
                i = 10
        super().__init__(field, to, field_name, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, parent_link=parent_link, on_delete=on_delete)
        self.multiple = False

class ManyToManyRel(ForeignObjectRel):
    """
    Used by ManyToManyField to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    def __init__(self, field, to, related_name=None, related_query_name=None, limit_choices_to=None, symmetrical=True, through=None, through_fields=None, db_constraint=True):
        if False:
            print('Hello World!')
        super().__init__(field, to, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to)
        if through and (not db_constraint):
            raise ValueError("Can't supply a through model and db_constraint=False")
        self.through = through
        if through_fields and (not through):
            raise ValueError('Cannot specify through_fields without a through model')
        self.through_fields = through_fields
        self.symmetrical = symmetrical
        self.db_constraint = db_constraint

    @property
    def identity(self):
        if False:
            while True:
                i = 10
        return super().identity + (self.through, make_hashable(self.through_fields), self.db_constraint)

    def get_related_field(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return the field in the 'to' object to which this relationship is tied.\n        Provided for symmetry with ManyToOneRel.\n        "
        opts = self.through._meta
        if self.through_fields:
            field = opts.get_field(self.through_fields[0])
        else:
            for field in opts.fields:
                rel = getattr(field, 'remote_field', None)
                if rel and rel.model == self.model:
                    break
        return field.foreign_related_fields[0]