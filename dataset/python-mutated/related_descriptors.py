"""
Accessors for related objects.

When a field defines a relation between two models, each model class provides
an attribute to access related instances of the other model class (unless the
reverse accessor has been disabled with related_name='+').

Accessors are implemented as descriptors in order to customize access and
assignment. This module defines the descriptor classes.

Forward accessors follow foreign keys. Reverse accessors trace them back. For
example, with the following models::

    class Parent(Model):
        pass

    class Child(Model):
        parent = ForeignKey(Parent, related_name='children')

 ``child.parent`` is a forward many-to-one relation. ``parent.children`` is a
reverse many-to-one relation.

There are three types of relations (many-to-one, one-to-one, and many-to-many)
and two directions (forward and reverse) for a total of six combinations.

1. Related instance on the forward side of a many-to-one relation:
   ``ForwardManyToOneDescriptor``.

   Uniqueness of foreign key values is irrelevant to accessing the related
   instance, making the many-to-one and one-to-one cases identical as far as
   the descriptor is concerned. The constraint is checked upstream (unicity
   validation in forms) or downstream (unique indexes in the database).

2. Related instance on the forward side of a one-to-one
   relation: ``ForwardOneToOneDescriptor``.

   It avoids querying the database when accessing the parent link field in
   a multi-table inheritance scenario.

3. Related instance on the reverse side of a one-to-one relation:
   ``ReverseOneToOneDescriptor``.

   One-to-one relations are asymmetrical, despite the apparent symmetry of the
   name, because they're implemented in the database with a foreign key from
   one table to another. As a consequence ``ReverseOneToOneDescriptor`` is
   slightly different from ``ForwardManyToOneDescriptor``.

4. Related objects manager for related instances on the reverse side of a
   many-to-one relation: ``ReverseManyToOneDescriptor``.

   Unlike the previous two classes, this one provides access to a collection
   of objects. It returns a manager rather than an instance.

5. Related objects manager for related instances on the forward or reverse
   sides of a many-to-many relation: ``ManyToManyDescriptor``.

   Many-to-many relations are symmetrical. The syntax of Django models
   requires declaring them on one side but that's an implementation detail.
   They could be declared on the other side without any change in behavior.
   Therefore the forward and reverse descriptors can be the same.

   If you're looking for ``ForwardManyToManyDescriptor`` or
   ``ReverseManyToManyDescriptor``, use ``ManyToManyDescriptor`` instead.
"""
import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections, router, transaction
from django.db.models import Q, Window, signals
from django.db.models.functions import RowNumber
from django.db.models.lookups import GreaterThan, LessThanOrEqual
from django.db.models.query import QuerySet
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData, resolve_callables
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property

class ForeignKeyDeferredAttribute(DeferredAttribute):

    def __set__(self, instance, value):
        if False:
            return 10
        if instance.__dict__.get(self.field.attname) != value and self.field.is_cached(instance):
            self.field.delete_cached_value(instance)
        instance.__dict__[self.field.attname] = value

def _filter_prefetch_queryset(queryset, field_name, instances):
    if False:
        return 10
    predicate = Q(**{f'{field_name}__in': instances})
    db = queryset._db or DEFAULT_DB_ALIAS
    if queryset.query.is_sliced:
        if not connections[db].features.supports_over_clause:
            raise NotSupportedError('Prefetching from a limited queryset is only supported on backends that support window functions.')
        (low_mark, high_mark) = (queryset.query.low_mark, queryset.query.high_mark)
        order_by = [expr for (expr, _) in queryset.query.get_compiler(using=db).get_order_by()]
        window = Window(RowNumber(), partition_by=field_name, order_by=order_by)
        predicate &= GreaterThan(window, low_mark)
        if high_mark is not None:
            predicate &= LessThanOrEqual(window, high_mark)
        queryset.query.clear_limits()
    return queryset.filter(predicate)

class ForwardManyToOneDescriptor:
    """
    Accessor to the related object on the forward side of a many-to-one or
    one-to-one (via ForwardOneToOneDescriptor subclass) relation.

    In the example::

        class Child(Model):
            parent = ForeignKey(Parent, related_name='children')

    ``Child.parent`` is a ``ForwardManyToOneDescriptor`` instance.
    """

    def __init__(self, field_with_rel):
        if False:
            i = 10
            return i + 15
        self.field = field_with_rel

    @cached_property
    def RelatedObjectDoesNotExist(self):
        if False:
            i = 10
            return i + 15
        return type('RelatedObjectDoesNotExist', (self.field.remote_field.model.DoesNotExist, AttributeError), {'__module__': self.field.model.__module__, '__qualname__': '%s.%s.RelatedObjectDoesNotExist' % (self.field.model.__qualname__, self.field.name)})

    def is_cached(self, instance):
        if False:
            return 10
        return self.field.is_cached(instance)

    def get_queryset(self, **hints):
        if False:
            for i in range(10):
                print('nop')
        return self.field.remote_field.model._base_manager.db_manager(hints=hints).all()

    def get_prefetch_queryset(self, instances, queryset=None):
        if False:
            return 10
        warnings.warn('get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() instead.', RemovedInDjango60Warning, stacklevel=2)
        if queryset is None:
            return self.get_prefetch_querysets(instances)
        return self.get_prefetch_querysets(instances, [queryset])

    def get_prefetch_querysets(self, instances, querysets=None):
        if False:
            for i in range(10):
                print('nop')
        if querysets and len(querysets) != 1:
            raise ValueError('querysets argument of get_prefetch_querysets() should have a length of 1.')
        queryset = querysets[0] if querysets else self.get_queryset()
        queryset._add_hints(instance=instances[0])
        rel_obj_attr = self.field.get_foreign_related_value
        instance_attr = self.field.get_local_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        related_field = self.field.foreign_related_fields[0]
        remote_field = self.field.remote_field
        if remote_field.is_hidden() or len(self.field.foreign_related_fields) == 1:
            query = {'%s__in' % related_field.name: {instance_attr(inst)[0] for inst in instances}}
        else:
            query = {'%s__in' % self.field.related_query_name(): instances}
        queryset = queryset.filter(**query)
        if not remote_field.multiple:
            for rel_obj in queryset:
                instance = instances_dict[rel_obj_attr(rel_obj)]
                remote_field.set_cached_value(rel_obj, instance)
        return (queryset, rel_obj_attr, instance_attr, True, self.field.get_cache_name(), False)

    def get_object(self, instance):
        if False:
            return 10
        qs = self.get_queryset(instance=instance)
        return qs.get(self.field.get_reverse_related_filter(instance))

    def __get__(self, instance, cls=None):
        if False:
            while True:
                i = 10
        "\n        Get the related instance through the forward relation.\n\n        With the example above, when getting ``child.parent``:\n\n        - ``self`` is the descriptor managing the ``parent`` attribute\n        - ``instance`` is the ``child`` instance\n        - ``cls`` is the ``Child`` class (we don't need it)\n        "
        if instance is None:
            return self
        try:
            rel_obj = self.field.get_cached_value(instance)
        except KeyError:
            has_value = None not in self.field.get_local_related_value(instance)
            ancestor_link = instance._meta.get_ancestor_link(self.field.model) if has_value else None
            if ancestor_link and ancestor_link.is_cached(instance):
                ancestor = ancestor_link.get_cached_value(instance)
                rel_obj = self.field.get_cached_value(ancestor, default=None)
            else:
                rel_obj = None
            if rel_obj is None and has_value:
                rel_obj = self.get_object(instance)
                remote_field = self.field.remote_field
                if not remote_field.multiple:
                    remote_field.set_cached_value(rel_obj, instance)
            self.field.set_cached_value(instance, rel_obj)
        if rel_obj is None and (not self.field.null):
            raise self.RelatedObjectDoesNotExist('%s has no %s.' % (self.field.model.__name__, self.field.name))
        else:
            return rel_obj

    def __set__(self, instance, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the related instance through the forward relation.\n\n        With the example above, when setting ``child.parent = parent``:\n\n        - ``self`` is the descriptor managing the ``parent`` attribute\n        - ``instance`` is the ``child`` instance\n        - ``value`` is the ``parent`` instance on the right of the equal sign\n        '
        if value is not None and (not isinstance(value, self.field.remote_field.model._meta.concrete_model)):
            raise ValueError('Cannot assign "%r": "%s.%s" must be a "%s" instance.' % (value, instance._meta.object_name, self.field.name, self.field.remote_field.model._meta.object_name))
        elif value is not None:
            if instance._state.db is None:
                instance._state.db = router.db_for_write(instance.__class__, instance=value)
            if value._state.db is None:
                value._state.db = router.db_for_write(value.__class__, instance=instance)
            if not router.allow_relation(value, instance):
                raise ValueError('Cannot assign "%r": the current database router prevents this relation.' % value)
        remote_field = self.field.remote_field
        if value is None:
            related = self.field.get_cached_value(instance, default=None)
            if related is not None:
                remote_field.set_cached_value(related, None)
            for (lh_field, rh_field) in self.field.related_fields:
                setattr(instance, lh_field.attname, None)
        else:
            for (lh_field, rh_field) in self.field.related_fields:
                setattr(instance, lh_field.attname, getattr(value, rh_field.attname))
        self.field.set_cached_value(instance, value)
        if value is not None and (not remote_field.multiple):
            remote_field.set_cached_value(value, instance)

    def __reduce__(self):
        if False:
            return 10
        '\n        Pickling should return the instance attached by self.field on the\n        model, not a new copy of that descriptor. Use getattr() to retrieve\n        the instance directly from the model.\n        '
        return (getattr, (self.field.model, self.field.name))

class ForwardOneToOneDescriptor(ForwardManyToOneDescriptor):
    """
    Accessor to the related object on the forward side of a one-to-one relation.

    In the example::

        class Restaurant(Model):
            place = OneToOneField(Place, related_name='restaurant')

    ``Restaurant.place`` is a ``ForwardOneToOneDescriptor`` instance.
    """

    def get_object(self, instance):
        if False:
            return 10
        if self.field.remote_field.parent_link:
            deferred = instance.get_deferred_fields()
            rel_model = self.field.remote_field.model
            fields = [field.attname for field in rel_model._meta.concrete_fields]
            if not any((field in fields for field in deferred)):
                kwargs = {field: getattr(instance, field) for field in fields}
                obj = rel_model(**kwargs)
                obj._state.adding = instance._state.adding
                obj._state.db = instance._state.db
                return obj
        return super().get_object(instance)

    def __set__(self, instance, value):
        if False:
            for i in range(10):
                print('nop')
        super().__set__(instance, value)
        if self.field.primary_key and self.field.remote_field.parent_link:
            opts = instance._meta
            inherited_pk_fields = [field for field in opts.concrete_fields if field.primary_key and field.remote_field]
            for field in inherited_pk_fields:
                rel_model_pk_name = field.remote_field.model._meta.pk.attname
                raw_value = getattr(value, rel_model_pk_name) if value is not None else None
                setattr(instance, rel_model_pk_name, raw_value)

class ReverseOneToOneDescriptor:
    """
    Accessor to the related object on the reverse side of a one-to-one
    relation.

    In the example::

        class Restaurant(Model):
            place = OneToOneField(Place, related_name='restaurant')

    ``Place.restaurant`` is a ``ReverseOneToOneDescriptor`` instance.
    """

    def __init__(self, related):
        if False:
            i = 10
            return i + 15
        self.related = related

    @cached_property
    def RelatedObjectDoesNotExist(self):
        if False:
            print('Hello World!')
        return type('RelatedObjectDoesNotExist', (self.related.related_model.DoesNotExist, AttributeError), {'__module__': self.related.model.__module__, '__qualname__': '%s.%s.RelatedObjectDoesNotExist' % (self.related.model.__qualname__, self.related.name)})

    def is_cached(self, instance):
        if False:
            print('Hello World!')
        return self.related.is_cached(instance)

    def get_queryset(self, **hints):
        if False:
            return 10
        return self.related.related_model._base_manager.db_manager(hints=hints).all()

    def get_prefetch_queryset(self, instances, queryset=None):
        if False:
            return 10
        warnings.warn('get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() instead.', RemovedInDjango60Warning, stacklevel=2)
        if queryset is None:
            return self.get_prefetch_querysets(instances)
        return self.get_prefetch_querysets(instances, [queryset])

    def get_prefetch_querysets(self, instances, querysets=None):
        if False:
            i = 10
            return i + 15
        if querysets and len(querysets) != 1:
            raise ValueError('querysets argument of get_prefetch_querysets() should have a length of 1.')
        queryset = querysets[0] if querysets else self.get_queryset()
        queryset._add_hints(instance=instances[0])
        rel_obj_attr = self.related.field.get_local_related_value
        instance_attr = self.related.field.get_foreign_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        query = {'%s__in' % self.related.field.name: instances}
        queryset = queryset.filter(**query)
        for rel_obj in queryset:
            instance = instances_dict[rel_obj_attr(rel_obj)]
            self.related.field.set_cached_value(rel_obj, instance)
        return (queryset, rel_obj_attr, instance_attr, True, self.related.get_cache_name(), False)

    def __get__(self, instance, cls=None):
        if False:
            i = 10
            return i + 15
        '\n        Get the related instance through the reverse relation.\n\n        With the example above, when getting ``place.restaurant``:\n\n        - ``self`` is the descriptor managing the ``restaurant`` attribute\n        - ``instance`` is the ``place`` instance\n        - ``cls`` is the ``Place`` class (unused)\n\n        Keep in mind that ``Restaurant`` holds the foreign key to ``Place``.\n        '
        if instance is None:
            return self
        try:
            rel_obj = self.related.get_cached_value(instance)
        except KeyError:
            related_pk = instance.pk
            if related_pk is None:
                rel_obj = None
            else:
                filter_args = self.related.field.get_forward_related_filter(instance)
                try:
                    rel_obj = self.get_queryset(instance=instance).get(**filter_args)
                except self.related.related_model.DoesNotExist:
                    rel_obj = None
                else:
                    self.related.field.set_cached_value(rel_obj, instance)
            self.related.set_cached_value(instance, rel_obj)
        if rel_obj is None:
            raise self.RelatedObjectDoesNotExist('%s has no %s.' % (instance.__class__.__name__, self.related.get_accessor_name()))
        else:
            return rel_obj

    def __set__(self, instance, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the related instance through the reverse relation.\n\n        With the example above, when setting ``place.restaurant = restaurant``:\n\n        - ``self`` is the descriptor managing the ``restaurant`` attribute\n        - ``instance`` is the ``place`` instance\n        - ``value`` is the ``restaurant`` instance on the right of the equal sign\n\n        Keep in mind that ``Restaurant`` holds the foreign key to ``Place``.\n        '
        if value is None:
            rel_obj = self.related.get_cached_value(instance, default=None)
            if rel_obj is not None:
                self.related.delete_cached_value(instance)
                setattr(rel_obj, self.related.field.name, None)
        elif not isinstance(value, self.related.related_model):
            raise ValueError('Cannot assign "%r": "%s.%s" must be a "%s" instance.' % (value, instance._meta.object_name, self.related.get_accessor_name(), self.related.related_model._meta.object_name))
        else:
            if instance._state.db is None:
                instance._state.db = router.db_for_write(instance.__class__, instance=value)
            if value._state.db is None:
                value._state.db = router.db_for_write(value.__class__, instance=instance)
            if not router.allow_relation(value, instance):
                raise ValueError('Cannot assign "%r": the current database router prevents this relation.' % value)
            related_pk = tuple((getattr(instance, field.attname) for field in self.related.field.foreign_related_fields))
            for (index, field) in enumerate(self.related.field.local_related_fields):
                setattr(value, field.attname, related_pk[index])
            self.related.set_cached_value(instance, value)
            self.related.field.set_cached_value(value, instance)

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (getattr, (self.related.model, self.related.name))

class ReverseManyToOneDescriptor:
    """
    Accessor to the related objects manager on the reverse side of a
    many-to-one relation.

    In the example::

        class Child(Model):
            parent = ForeignKey(Parent, related_name='children')

    ``Parent.children`` is a ``ReverseManyToOneDescriptor`` instance.

    Most of the implementation is delegated to a dynamically defined manager
    class built by ``create_forward_many_to_many_manager()`` defined below.
    """

    def __init__(self, rel):
        if False:
            for i in range(10):
                print('nop')
        self.rel = rel
        self.field = rel.field

    @cached_property
    def related_manager_cls(self):
        if False:
            for i in range(10):
                print('nop')
        related_model = self.rel.related_model
        return create_reverse_many_to_one_manager(related_model._default_manager.__class__, self.rel)

    def __get__(self, instance, cls=None):
        if False:
            print('Hello World!')
        '\n        Get the related objects through the reverse relation.\n\n        With the example above, when getting ``parent.children``:\n\n        - ``self`` is the descriptor managing the ``children`` attribute\n        - ``instance`` is the ``parent`` instance\n        - ``cls`` is the ``Parent`` class (unused)\n        '
        if instance is None:
            return self
        return self.related_manager_cls(instance)

    def _get_set_deprecation_msg_params(self):
        if False:
            return 10
        return ('reverse side of a related set', self.rel.get_accessor_name())

    def __set__(self, instance, value):
        if False:
            i = 10
            return i + 15
        raise TypeError('Direct assignment to the %s is prohibited. Use %s.set() instead.' % self._get_set_deprecation_msg_params())

def create_reverse_many_to_one_manager(superclass, rel):
    if False:
        while True:
            i = 10
    '\n    Create a manager for the reverse side of a many-to-one relation.\n\n    This manager subclasses another manager, generally the default manager of\n    the related model, and adds behaviors specific to many-to-one relations.\n    '

    class RelatedManager(superclass, AltersData):

        def __init__(self, instance):
            if False:
                print('Hello World!')
            super().__init__()
            self.instance = instance
            self.model = rel.related_model
            self.field = rel.field
            self.core_filters = {self.field.name: instance}

        def __call__(self, *, manager):
            if False:
                return 10
            manager = getattr(self.model, manager)
            manager_class = create_reverse_many_to_one_manager(manager.__class__, rel)
            return manager_class(self.instance)
        do_not_call_in_templates = True

        def _check_fk_val(self):
            if False:
                return 10
            for field in self.field.foreign_related_fields:
                if getattr(self.instance, field.attname) is None:
                    raise ValueError(f'"{self.instance!r}" needs to have a value for field "{field.attname}" before this relationship can be used.')

        def _apply_rel_filters(self, queryset):
            if False:
                return 10
            '\n            Filter the queryset for the instance this manager is bound to.\n            '
            db = self._db or router.db_for_read(self.model, instance=self.instance)
            empty_strings_as_null = connections[db].features.interprets_empty_strings_as_nulls
            queryset._add_hints(instance=self.instance)
            if self._db:
                queryset = queryset.using(self._db)
            queryset._defer_next_filter = True
            queryset = queryset.filter(**self.core_filters)
            for field in self.field.foreign_related_fields:
                val = getattr(self.instance, field.attname)
                if val is None or (val == '' and empty_strings_as_null):
                    return queryset.none()
            if self.field.many_to_one:
                try:
                    target_field = self.field.target_field
                except FieldError:
                    rel_obj_id = tuple([getattr(self.instance, target_field.attname) for target_field in self.field.path_infos[-1].target_fields])
                else:
                    rel_obj_id = getattr(self.instance, target_field.attname)
                queryset._known_related_objects = {self.field: {rel_obj_id: self.instance}}
            return queryset

        def _remove_prefetched_objects(self):
            if False:
                while True:
                    i = 10
            try:
                self.instance._prefetched_objects_cache.pop(self.field.remote_field.get_cache_name())
            except (AttributeError, KeyError):
                pass

        def get_queryset(self):
            if False:
                i = 10
                return i + 15
            if self.instance.pk is None:
                raise ValueError(f'{self.instance.__class__.__name__!r} instance needs to have a primary key value before this relationship can be used.')
            try:
                return self.instance._prefetched_objects_cache[self.field.remote_field.get_cache_name()]
            except (AttributeError, KeyError):
                queryset = super().get_queryset()
                return self._apply_rel_filters(queryset)

        def get_prefetch_queryset(self, instances, queryset=None):
            if False:
                for i in range(10):
                    print('nop')
            warnings.warn('get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() instead.', RemovedInDjango60Warning, stacklevel=2)
            if queryset is None:
                return self.get_prefetch_querysets(instances)
            return self.get_prefetch_querysets(instances, [queryset])

        def get_prefetch_querysets(self, instances, querysets=None):
            if False:
                return 10
            if querysets and len(querysets) != 1:
                raise ValueError('querysets argument of get_prefetch_querysets() should have a length of 1.')
            queryset = querysets[0] if querysets else super().get_queryset()
            queryset._add_hints(instance=instances[0])
            queryset = queryset.using(queryset._db or self._db)
            rel_obj_attr = self.field.get_local_related_value
            instance_attr = self.field.get_foreign_related_value
            instances_dict = {instance_attr(inst): inst for inst in instances}
            queryset = _filter_prefetch_queryset(queryset, self.field.name, instances)
            for rel_obj in queryset:
                if not self.field.is_cached(rel_obj):
                    instance = instances_dict[rel_obj_attr(rel_obj)]
                    setattr(rel_obj, self.field.name, instance)
            cache_name = self.field.remote_field.get_cache_name()
            return (queryset, rel_obj_attr, instance_attr, False, cache_name, False)

        def add(self, *objs, bulk=True):
            if False:
                i = 10
                return i + 15
            self._check_fk_val()
            self._remove_prefetched_objects()
            db = router.db_for_write(self.model, instance=self.instance)

            def check_and_update_obj(obj):
                if False:
                    i = 10
                    return i + 15
                if not isinstance(obj, self.model):
                    raise TypeError("'%s' instance expected, got %r" % (self.model._meta.object_name, obj))
                setattr(obj, self.field.name, self.instance)
            if bulk:
                pks = []
                for obj in objs:
                    check_and_update_obj(obj)
                    if obj._state.adding or obj._state.db != db:
                        raise ValueError("%r instance isn't saved. Use bulk=False or save the object first." % obj)
                    pks.append(obj.pk)
                self.model._base_manager.using(db).filter(pk__in=pks).update(**{self.field.name: self.instance})
            else:
                with transaction.atomic(using=db, savepoint=False):
                    for obj in objs:
                        check_and_update_obj(obj)
                        obj.save()
        add.alters_data = True

        async def aadd(self, *objs, bulk=True):
            return await sync_to_async(self.add)(*objs, bulk=bulk)
        aadd.alters_data = True

        def create(self, **kwargs):
            if False:
                while True:
                    i = 10
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).create(**kwargs)
        create.alters_data = True

        async def acreate(self, **kwargs):
            return await sync_to_async(self.create)(**kwargs)
        acreate.alters_data = True

        def get_or_create(self, **kwargs):
            if False:
                return 10
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).get_or_create(**kwargs)
        get_or_create.alters_data = True

        async def aget_or_create(self, **kwargs):
            return await sync_to_async(self.get_or_create)(**kwargs)
        aget_or_create.alters_data = True

        def update_or_create(self, **kwargs):
            if False:
                while True:
                    i = 10
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).update_or_create(**kwargs)
        update_or_create.alters_data = True

        async def aupdate_or_create(self, **kwargs):
            return await sync_to_async(self.update_or_create)(**kwargs)
        aupdate_or_create.alters_data = True
        if rel.field.null:

            def remove(self, *objs, bulk=True):
                if False:
                    i = 10
                    return i + 15
                if not objs:
                    return
                self._check_fk_val()
                val = self.field.get_foreign_related_value(self.instance)
                old_ids = set()
                for obj in objs:
                    if not isinstance(obj, self.model):
                        raise TypeError("'%s' instance expected, got %r" % (self.model._meta.object_name, obj))
                    if self.field.get_local_related_value(obj) == val:
                        old_ids.add(obj.pk)
                    else:
                        raise self.field.remote_field.model.DoesNotExist('%r is not related to %r.' % (obj, self.instance))
                self._clear(self.filter(pk__in=old_ids), bulk)
            remove.alters_data = True

            async def aremove(self, *objs, bulk=True):
                return await sync_to_async(self.remove)(*objs, bulk=bulk)
            aremove.alters_data = True

            def clear(self, *, bulk=True):
                if False:
                    for i in range(10):
                        print('nop')
                self._check_fk_val()
                self._clear(self, bulk)
            clear.alters_data = True

            async def aclear(self, *, bulk=True):
                return await sync_to_async(self.clear)(bulk=bulk)
            aclear.alters_data = True

            def _clear(self, queryset, bulk):
                if False:
                    print('Hello World!')
                self._remove_prefetched_objects()
                db = router.db_for_write(self.model, instance=self.instance)
                queryset = queryset.using(db)
                if bulk:
                    queryset.update(**{self.field.name: None})
                else:
                    with transaction.atomic(using=db, savepoint=False):
                        for obj in queryset:
                            setattr(obj, self.field.name, None)
                            obj.save(update_fields=[self.field.name])
            _clear.alters_data = True

        def set(self, objs, *, bulk=True, clear=False):
            if False:
                return 10
            self._check_fk_val()
            objs = tuple(objs)
            if self.field.null:
                db = router.db_for_write(self.model, instance=self.instance)
                with transaction.atomic(using=db, savepoint=False):
                    if clear:
                        self.clear(bulk=bulk)
                        self.add(*objs, bulk=bulk)
                    else:
                        old_objs = set(self.using(db).all())
                        new_objs = []
                        for obj in objs:
                            if obj in old_objs:
                                old_objs.remove(obj)
                            else:
                                new_objs.append(obj)
                        self.remove(*old_objs, bulk=bulk)
                        self.add(*new_objs, bulk=bulk)
            else:
                self.add(*objs, bulk=bulk)
        set.alters_data = True

        async def aset(self, objs, *, bulk=True, clear=False):
            return await sync_to_async(self.set)(objs=objs, bulk=bulk, clear=clear)
        aset.alters_data = True
    return RelatedManager

class ManyToManyDescriptor(ReverseManyToOneDescriptor):
    """
    Accessor to the related objects manager on the forward and reverse sides of
    a many-to-many relation.

    In the example::

        class Pizza(Model):
            toppings = ManyToManyField(Topping, related_name='pizzas')

    ``Pizza.toppings`` and ``Topping.pizzas`` are ``ManyToManyDescriptor``
    instances.

    Most of the implementation is delegated to a dynamically defined manager
    class built by ``create_forward_many_to_many_manager()`` defined below.
    """

    def __init__(self, rel, reverse=False):
        if False:
            while True:
                i = 10
        super().__init__(rel)
        self.reverse = reverse

    @property
    def through(self):
        if False:
            i = 10
            return i + 15
        return self.rel.through

    @cached_property
    def related_manager_cls(self):
        if False:
            for i in range(10):
                print('nop')
        related_model = self.rel.related_model if self.reverse else self.rel.model
        return create_forward_many_to_many_manager(related_model._default_manager.__class__, self.rel, reverse=self.reverse)

    def _get_set_deprecation_msg_params(self):
        if False:
            return 10
        return ('%s side of a many-to-many set' % ('reverse' if self.reverse else 'forward'), self.rel.get_accessor_name() if self.reverse else self.field.name)

def create_forward_many_to_many_manager(superclass, rel, reverse):
    if False:
        i = 10
        return i + 15
    '\n    Create a manager for the either side of a many-to-many relation.\n\n    This manager subclasses another manager, generally the default manager of\n    the related model, and adds behaviors specific to many-to-many relations.\n    '

    class ManyRelatedManager(superclass, AltersData):

        def __init__(self, instance=None):
            if False:
                return 10
            super().__init__()
            self.instance = instance
            if not reverse:
                self.model = rel.model
                self.query_field_name = rel.field.related_query_name()
                self.prefetch_cache_name = rel.field.name
                self.source_field_name = rel.field.m2m_field_name()
                self.target_field_name = rel.field.m2m_reverse_field_name()
                self.symmetrical = rel.symmetrical
            else:
                self.model = rel.related_model
                self.query_field_name = rel.field.name
                self.prefetch_cache_name = rel.field.related_query_name()
                self.source_field_name = rel.field.m2m_reverse_field_name()
                self.target_field_name = rel.field.m2m_field_name()
                self.symmetrical = False
            self.through = rel.through
            self.reverse = reverse
            self.source_field = self.through._meta.get_field(self.source_field_name)
            self.target_field = self.through._meta.get_field(self.target_field_name)
            self.core_filters = {}
            self.pk_field_names = {}
            for (lh_field, rh_field) in self.source_field.related_fields:
                core_filter_key = '%s__%s' % (self.query_field_name, rh_field.name)
                self.core_filters[core_filter_key] = getattr(instance, rh_field.attname)
                self.pk_field_names[lh_field.name] = rh_field.name
            self.related_val = self.source_field.get_foreign_related_value(instance)
            if None in self.related_val:
                raise ValueError('"%r" needs to have a value for field "%s" before this many-to-many relationship can be used.' % (instance, self.pk_field_names[self.source_field_name]))
            if instance.pk is None:
                raise ValueError('%r instance needs to have a primary key value before a many-to-many relationship can be used.' % instance.__class__.__name__)

        def __call__(self, *, manager):
            if False:
                i = 10
                return i + 15
            manager = getattr(self.model, manager)
            manager_class = create_forward_many_to_many_manager(manager.__class__, rel, reverse)
            return manager_class(instance=self.instance)
        do_not_call_in_templates = True

        def _build_remove_filters(self, removed_vals):
            if False:
                i = 10
                return i + 15
            filters = Q.create([(self.source_field_name, self.related_val)])
            removed_vals_filters = not isinstance(removed_vals, QuerySet) or removed_vals._has_filters()
            if removed_vals_filters:
                filters &= Q.create([(f'{self.target_field_name}__in', removed_vals)])
            if self.symmetrical:
                symmetrical_filters = Q.create([(self.target_field_name, self.related_val)])
                if removed_vals_filters:
                    symmetrical_filters &= Q.create([(f'{self.source_field_name}__in', removed_vals)])
                filters |= symmetrical_filters
            return filters

        def _apply_rel_filters(self, queryset):
            if False:
                print('Hello World!')
            '\n            Filter the queryset for the instance this manager is bound to.\n            '
            queryset._add_hints(instance=self.instance)
            if self._db:
                queryset = queryset.using(self._db)
            queryset._defer_next_filter = True
            return queryset._next_is_sticky().filter(**self.core_filters)

        def _remove_prefetched_objects(self):
            if False:
                while True:
                    i = 10
            try:
                self.instance._prefetched_objects_cache.pop(self.prefetch_cache_name)
            except (AttributeError, KeyError):
                pass

        def get_queryset(self):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return self.instance._prefetched_objects_cache[self.prefetch_cache_name]
            except (AttributeError, KeyError):
                queryset = super().get_queryset()
                return self._apply_rel_filters(queryset)

        def get_prefetch_queryset(self, instances, queryset=None):
            if False:
                for i in range(10):
                    print('nop')
            warnings.warn('get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() instead.', RemovedInDjango60Warning, stacklevel=2)
            if queryset is None:
                return self.get_prefetch_querysets(instances)
            return self.get_prefetch_querysets(instances, [queryset])

        def get_prefetch_querysets(self, instances, querysets=None):
            if False:
                print('Hello World!')
            if querysets and len(querysets) != 1:
                raise ValueError('querysets argument of get_prefetch_querysets() should have a length of 1.')
            queryset = querysets[0] if querysets else super().get_queryset()
            queryset._add_hints(instance=instances[0])
            queryset = queryset.using(queryset._db or self._db)
            queryset = _filter_prefetch_queryset(queryset._next_is_sticky(), self.query_field_name, instances)
            fk = self.through._meta.get_field(self.source_field_name)
            join_table = fk.model._meta.db_table
            connection = connections[queryset.db]
            qn = connection.ops.quote_name
            queryset = queryset.extra(select={'_prefetch_related_val_%s' % f.attname: '%s.%s' % (qn(join_table), qn(f.column)) for f in fk.local_related_fields})
            return (queryset, lambda result: tuple((f.get_db_prep_value(getattr(result, f'_prefetch_related_val_{f.attname}'), connection) for f in fk.local_related_fields)), lambda inst: tuple((f.get_db_prep_value(getattr(inst, f.attname), connection) for f in fk.foreign_related_fields)), False, self.prefetch_cache_name, False)

        def add(self, *objs, through_defaults=None):
            if False:
                i = 10
                return i + 15
            self._remove_prefetched_objects()
            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                self._add_items(self.source_field_name, self.target_field_name, *objs, through_defaults=through_defaults)
                if self.symmetrical:
                    self._add_items(self.target_field_name, self.source_field_name, *objs, through_defaults=through_defaults)
        add.alters_data = True

        async def aadd(self, *objs, through_defaults=None):
            return await sync_to_async(self.add)(*objs, through_defaults=through_defaults)
        aadd.alters_data = True

        def remove(self, *objs):
            if False:
                i = 10
                return i + 15
            self._remove_prefetched_objects()
            self._remove_items(self.source_field_name, self.target_field_name, *objs)
        remove.alters_data = True

        async def aremove(self, *objs):
            return await sync_to_async(self.remove)(*objs)
        aremove.alters_data = True

        def clear(self):
            if False:
                return 10
            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                signals.m2m_changed.send(sender=self.through, action='pre_clear', instance=self.instance, reverse=self.reverse, model=self.model, pk_set=None, using=db)
                self._remove_prefetched_objects()
                filters = self._build_remove_filters(super().get_queryset().using(db))
                self.through._default_manager.using(db).filter(filters).delete()
                signals.m2m_changed.send(sender=self.through, action='post_clear', instance=self.instance, reverse=self.reverse, model=self.model, pk_set=None, using=db)
        clear.alters_data = True

        async def aclear(self):
            return await sync_to_async(self.clear)()
        aclear.alters_data = True

        def set(self, objs, *, clear=False, through_defaults=None):
            if False:
                i = 10
                return i + 15
            objs = tuple(objs)
            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                if clear:
                    self.clear()
                    self.add(*objs, through_defaults=through_defaults)
                else:
                    old_ids = set(self.using(db).values_list(self.target_field.target_field.attname, flat=True))
                    new_objs = []
                    for obj in objs:
                        fk_val = self.target_field.get_foreign_related_value(obj)[0] if isinstance(obj, self.model) else self.target_field.get_prep_value(obj)
                        if fk_val in old_ids:
                            old_ids.remove(fk_val)
                        else:
                            new_objs.append(obj)
                    self.remove(*old_ids)
                    self.add(*new_objs, through_defaults=through_defaults)
        set.alters_data = True

        async def aset(self, objs, *, clear=False, through_defaults=None):
            return await sync_to_async(self.set)(objs=objs, clear=clear, through_defaults=through_defaults)
        aset.alters_data = True

        def create(self, *, through_defaults=None, **kwargs):
            if False:
                print('Hello World!')
            db = router.db_for_write(self.instance.__class__, instance=self.instance)
            new_obj = super(ManyRelatedManager, self.db_manager(db)).create(**kwargs)
            self.add(new_obj, through_defaults=through_defaults)
            return new_obj
        create.alters_data = True

        async def acreate(self, *, through_defaults=None, **kwargs):
            return await sync_to_async(self.create)(through_defaults=through_defaults, **kwargs)
        acreate.alters_data = True

        def get_or_create(self, *, through_defaults=None, **kwargs):
            if False:
                i = 10
                return i + 15
            db = router.db_for_write(self.instance.__class__, instance=self.instance)
            (obj, created) = super(ManyRelatedManager, self.db_manager(db)).get_or_create(**kwargs)
            if created:
                self.add(obj, through_defaults=through_defaults)
            return (obj, created)
        get_or_create.alters_data = True

        async def aget_or_create(self, *, through_defaults=None, **kwargs):
            return await sync_to_async(self.get_or_create)(through_defaults=through_defaults, **kwargs)
        aget_or_create.alters_data = True

        def update_or_create(self, *, through_defaults=None, **kwargs):
            if False:
                while True:
                    i = 10
            db = router.db_for_write(self.instance.__class__, instance=self.instance)
            (obj, created) = super(ManyRelatedManager, self.db_manager(db)).update_or_create(**kwargs)
            if created:
                self.add(obj, through_defaults=through_defaults)
            return (obj, created)
        update_or_create.alters_data = True

        async def aupdate_or_create(self, *, through_defaults=None, **kwargs):
            return await sync_to_async(self.update_or_create)(through_defaults=through_defaults, **kwargs)
        aupdate_or_create.alters_data = True

        def _get_target_ids(self, target_field_name, objs):
            if False:
                return 10
            '\n            Return the set of ids of `objs` that the target field references.\n            '
            from django.db.models import Model
            target_ids = set()
            target_field = self.through._meta.get_field(target_field_name)
            for obj in objs:
                if isinstance(obj, self.model):
                    if not router.allow_relation(obj, self.instance):
                        raise ValueError('Cannot add "%r": instance is on database "%s", value is on database "%s"' % (obj, self.instance._state.db, obj._state.db))
                    target_id = target_field.get_foreign_related_value(obj)[0]
                    if target_id is None:
                        raise ValueError('Cannot add "%r": the value for field "%s" is None' % (obj, target_field_name))
                    target_ids.add(target_id)
                elif isinstance(obj, Model):
                    raise TypeError("'%s' instance expected, got %r" % (self.model._meta.object_name, obj))
                else:
                    target_ids.add(target_field.get_prep_value(obj))
            return target_ids

        def _get_missing_target_ids(self, source_field_name, target_field_name, db, target_ids):
            if False:
                print('Hello World!')
            "\n            Return the subset of ids of `objs` that aren't already assigned to\n            this relationship.\n            "
            vals = self.through._default_manager.using(db).values_list(target_field_name, flat=True).filter(**{source_field_name: self.related_val[0], '%s__in' % target_field_name: target_ids})
            return target_ids.difference(vals)

        def _get_add_plan(self, db, source_field_name):
            if False:
                print('Hello World!')
            '\n            Return a boolean triple of the way the add should be performed.\n\n            The first element is whether or not bulk_create(ignore_conflicts)\n            can be used, the second whether or not signals must be sent, and\n            the third element is whether or not the immediate bulk insertion\n            with conflicts ignored can be performed.\n            '
            can_ignore_conflicts = self.through._meta.auto_created is not False and connections[db].features.supports_ignore_conflicts
            must_send_signals = (self.reverse or source_field_name == self.source_field_name) and signals.m2m_changed.has_listeners(self.through)
            return (can_ignore_conflicts, must_send_signals, can_ignore_conflicts and (not must_send_signals))

        def _add_items(self, source_field_name, target_field_name, *objs, through_defaults=None):
            if False:
                return 10
            if not objs:
                return
            through_defaults = dict(resolve_callables(through_defaults or {}))
            target_ids = self._get_target_ids(target_field_name, objs)
            db = router.db_for_write(self.through, instance=self.instance)
            (can_ignore_conflicts, must_send_signals, can_fast_add) = self._get_add_plan(db, source_field_name)
            if can_fast_add:
                self.through._default_manager.using(db).bulk_create([self.through(**{'%s_id' % source_field_name: self.related_val[0], '%s_id' % target_field_name: target_id}) for target_id in target_ids], ignore_conflicts=True)
                return
            missing_target_ids = self._get_missing_target_ids(source_field_name, target_field_name, db, target_ids)
            with transaction.atomic(using=db, savepoint=False):
                if must_send_signals:
                    signals.m2m_changed.send(sender=self.through, action='pre_add', instance=self.instance, reverse=self.reverse, model=self.model, pk_set=missing_target_ids, using=db)
                self.through._default_manager.using(db).bulk_create([self.through(**through_defaults, **{'%s_id' % source_field_name: self.related_val[0], '%s_id' % target_field_name: target_id}) for target_id in missing_target_ids], ignore_conflicts=can_ignore_conflicts)
                if must_send_signals:
                    signals.m2m_changed.send(sender=self.through, action='post_add', instance=self.instance, reverse=self.reverse, model=self.model, pk_set=missing_target_ids, using=db)

        def _remove_items(self, source_field_name, target_field_name, *objs):
            if False:
                while True:
                    i = 10
            if not objs:
                return
            old_ids = set()
            for obj in objs:
                if isinstance(obj, self.model):
                    fk_val = self.target_field.get_foreign_related_value(obj)[0]
                    old_ids.add(fk_val)
                else:
                    old_ids.add(obj)
            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                signals.m2m_changed.send(sender=self.through, action='pre_remove', instance=self.instance, reverse=self.reverse, model=self.model, pk_set=old_ids, using=db)
                target_model_qs = super().get_queryset()
                if target_model_qs._has_filters():
                    old_vals = target_model_qs.using(db).filter(**{'%s__in' % self.target_field.target_field.attname: old_ids})
                else:
                    old_vals = old_ids
                filters = self._build_remove_filters(old_vals)
                self.through._default_manager.using(db).filter(filters).delete()
                signals.m2m_changed.send(sender=self.through, action='post_remove', instance=self.instance, reverse=self.reverse, model=self.model, pk_set=old_ids, using=db)
    return ManyRelatedManager