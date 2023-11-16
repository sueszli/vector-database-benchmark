from django.db import models
from django.db.models.fields.related import ReverseManyToOneDescriptor
from django.db.models.lookups import StartsWith
from django.db.models.query_utils import PathInfo

class CustomForeignObjectRel(models.ForeignObjectRel):
    """
    Define some extra Field methods so this Rel acts more like a Field, which
    lets us use ReverseManyToOneDescriptor in both directions.
    """

    @property
    def foreign_related_fields(self):
        if False:
            return 10
        return tuple((lhs_field for (lhs_field, rhs_field) in self.field.related_fields))

    def get_attname(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

class StartsWithRelation(models.ForeignObject):
    """
    A ForeignObject that uses StartsWith operator in its joins instead of
    the default equality operator. This is logically a many-to-many relation
    and creates a ReverseManyToOneDescriptor in both directions.
    """
    auto_created = False
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False
    rel_class = CustomForeignObjectRel

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['on_delete'] = models.DO_NOTHING
        super().__init__(*args, **kwargs)

    @property
    def field(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Makes ReverseManyToOneDescriptor work in both directions.\n        '
        return self.remote_field

    def get_extra_restriction(self, alias, related_alias):
        if False:
            print('Hello World!')
        to_field = self.remote_field.model._meta.get_field(self.to_fields[0])
        from_field = self.model._meta.get_field(self.from_fields[0])
        return StartsWith(to_field.get_col(alias), from_field.get_col(related_alias))

    def get_joining_fields(self, reverse_join=False):
        if False:
            for i in range(10):
                print('nop')
        return ()

    def get_path_info(self, filtered_relation=None):
        if False:
            while True:
                i = 10
        to_opts = self.remote_field.model._meta
        from_opts = self.model._meta
        return [PathInfo(from_opts=from_opts, to_opts=to_opts, target_fields=(to_opts.pk,), join_field=self, m2m=False, direct=False, filtered_relation=filtered_relation)]

    def get_reverse_path_info(self, filtered_relation=None):
        if False:
            while True:
                i = 10
        to_opts = self.model._meta
        from_opts = self.remote_field.model._meta
        return [PathInfo(from_opts=from_opts, to_opts=to_opts, target_fields=(to_opts.pk,), join_field=self.remote_field, m2m=False, direct=False, filtered_relation=filtered_relation)]

    def contribute_to_class(self, cls, name, private_only=False):
        if False:
            i = 10
            return i + 15
        super().contribute_to_class(cls, name, private_only)
        setattr(cls, self.name, ReverseManyToOneDescriptor(self))

class BrokenContainsRelation(StartsWithRelation):
    """
    This model is designed to yield no join conditions and
    raise an exception in ``Join.as_sql()``.
    """

    def get_extra_restriction(self, alias, related_alias):
        if False:
            while True:
                i = 10
        return None

class SlugPage(models.Model):
    slug = models.CharField(max_length=20, unique=True)
    descendants = StartsWithRelation('self', from_fields=['slug'], to_fields=['slug'], related_name='ascendants')
    containers = BrokenContainsRelation('self', from_fields=['slug'], to_fields=['slug'])

    class Meta:
        ordering = ['slug']

    def __str__(self):
        if False:
            return 10
        return 'SlugPage %s' % self.slug