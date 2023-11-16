import uuid
from django.contrib.contenttypes.fields import GenericForeignKey, GenericRel
from django.contrib.contenttypes.models import ContentType
from django.db import connection, models
from django.utils.functional import cached_property
from django.utils.text import capfirst
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from modelcluster.models import ClusterableModel, get_all_child_relations
from taggit.models import ItemBase
from wagtail.blocks import StreamBlock
from wagtail.fields import StreamField

class ReferenceGroups:
    """
    Groups records in a ReferenceIndex queryset by their source object.

    Args:
        qs: (QuerySet[ReferenceIndex]) A QuerySet on the ReferenceIndex model

    Yields:
        A tuple (source_object, references) for each source object that appears
        in the queryset. source_object is the model instance of the source object
        and references is a list of references that occur in the QuerySet from
        that source object.
    """

    def __init__(self, qs):
        if False:
            print('Hello World!')
        self.qs = qs.order_by('base_content_type', 'object_id')

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        reference_fk = None
        references = []
        for reference in self.qs:
            if reference_fk != (reference.base_content_type_id, reference.object_id):
                if reference_fk is not None:
                    content_type = ContentType.objects.get_for_id(reference_fk[0])
                    object = content_type.get_object_for_this_type(pk=reference_fk[1])
                    yield (object, references)
                    references = []
                reference_fk = (reference.base_content_type_id, reference.object_id)
            references.append(reference)
        if references:
            content_type = ContentType.objects.get_for_id(reference_fk[0])
            object = content_type.get_object_for_this_type(pk=reference_fk[1])
            yield (object, references)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._count

    @cached_property
    def _count(self):
        if False:
            print('Hello World!')
        return self.qs.values('base_content_type', 'object_id').distinct().count()

    @cached_property
    def is_protected(self):
        if False:
            i = 10
            return i + 15
        return any((reference.on_delete == models.PROTECT for reference in self.qs))

    def count(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of rows that will be returned by iterating this\n        ReferenceGroups.\n\n        Just calls len(self) internally, this method only exists to allow\n        instances of this class to be used in a Paginator.\n        '
        return len(self)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return list(self)[key]

class ReferenceIndexQuerySet(models.QuerySet):

    def group_by_source_object(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a ReferenceGroups object for this queryset that will yield\n        references grouped by their source instance.\n        '
        return ReferenceGroups(self)

class ReferenceIndex(models.Model):
    """
    Records references between objects for quick retrieval of object usage.

    References are extracted from Foreign Keys, Chooser Blocks in StreamFields, and links in Rich Text Fields.
    This index allows us to efficiently find all of the references to a particular object from all of these sources.
    """
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, related_name='+')
    base_content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, related_name='+')
    object_id = models.CharField(max_length=255, verbose_name=_('object id'))
    to_content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, related_name='+')
    to_object_id = models.CharField(max_length=255, verbose_name=_('object id'))
    model_path = models.TextField()
    content_path = models.TextField()
    content_path_hash = models.UUIDField()
    objects = ReferenceIndexQuerySet.as_manager()
    wagtail_reference_index_ignore = True
    tracked_models = set()
    indexed_models = set()

    class Meta:
        unique_together = [('base_content_type', 'object_id', 'to_content_type', 'to_object_id', 'content_path_hash')]

    @classmethod
    def _get_base_content_type(cls, model_or_object):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the ContentType record that represents the base model of the\n        given model or object.\n\n        For a model that uses multi-table-inheritance, this returns the model\n        that contains the primary key. For example, for any page object, this\n        will return the content type of the Page model.\n        '
        parents = model_or_object._meta.get_parent_list()
        if parents:
            return ContentType.objects.get_for_model(parents[-1], for_concrete_model=False)
        else:
            return ContentType.objects.get_for_model(model_or_object, for_concrete_model=False)

    @classmethod
    def model_is_indexable(cls, model, allow_child_models=False):
        if False:
            print('Hello World!')
        '\n        Returns True if the given model may have outbound references that we would be interested in recording in the index.\n\n\n        Args:\n            model (type): a Django model class\n            allow_child_models (boolean): Child models are not indexable on their own. If you are looking at\n                                          a child model from the perspective of indexing it through its parent,\n                                          set this to True to disable checking for this. Default False.\n        '
        if getattr(model, 'wagtail_reference_index_ignore', False):
            return False
        if not allow_child_models and any((isinstance(field, ParentalKey) for field in model._meta.get_fields())):
            return False
        for field in model._meta.get_fields():
            if field.is_relation and field.many_to_one:
                if getattr(field, 'wagtail_reference_index_ignore', False):
                    continue
                if getattr(field.related_model, 'wagtail_reference_index_ignore', False):
                    continue
                if isinstance(field, (ParentalKey, GenericRel)):
                    continue
                return True
            if hasattr(field, 'extract_references'):
                return True
        if issubclass(model, ClusterableModel):
            for child_relation in get_all_child_relations(model):
                if cls.model_is_indexable(child_relation.related_model, allow_child_models=True):
                    return True
        return False

    @classmethod
    def register_model(cls, model):
        if False:
            while True:
                i = 10
        '\n        Registers the model for indexing.\n        '
        if model in cls.indexed_models:
            return
        if cls.model_is_indexable(model):
            cls.indexed_models.add(model)
            cls._register_as_tracked_model(model)

    @classmethod
    def _register_as_tracked_model(cls, model):
        if False:
            while True:
                i = 10
        '\n        Add the model and all of its ParentalKey-linked children to the set of\n        models to be tracked by signal handlers.\n        '
        if model in cls.tracked_models:
            return
        from wagtail.signal_handlers import connect_reference_index_signal_handlers_for_model
        cls.tracked_models.add(model)
        connect_reference_index_signal_handlers_for_model(model)
        for child_relation in get_all_child_relations(model):
            if cls.model_is_indexable(child_relation.related_model, allow_child_models=True):
                cls._register_as_tracked_model(child_relation.related_model)

    @classmethod
    def is_indexed(cls, model):
        if False:
            print('Hello World!')
        return model in cls.indexed_models

    @classmethod
    def _extract_references_from_object(cls, object):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generator that scans the given object and yields any references it finds.\n\n        Args:\n            object (Model): an instance of a Django model to scan for references\n\n        Yields:\n            A tuple (content_type_id, object_id, model_path, content_path) for each\n            reference found.\n\n            content_type_id (int): The ID of the ContentType record representing\n                                   the model of the referenced object\n\n            object_id (str): The primary key of the referenced object, converted\n                             to a string\n\n            model_path (str): The path to the field on the model of the source\n                              object where the reference was found\n\n            content_path (str): The path to the piece of content on the source\n                                object instance where the reference was found\n        '
        for field in object._meta.get_fields():
            if field.is_relation and field.many_to_one:
                if getattr(field, 'wagtail_reference_index_ignore', False):
                    continue
                if getattr(field.related_model, 'wagtail_reference_index_ignore', False):
                    continue
                if isinstance(field, (ParentalKey, GenericRel)):
                    continue
                if isinstance(field, GenericForeignKey):
                    ct_field = object._meta.get_field(field.ct_field)
                    fk_field = object._meta.get_field(field.fk_field)
                    ct_value = ct_field.value_from_object(object)
                    fk_value = fk_field.value_from_object(object)
                    if ct_value is not None and fk_value is not None:
                        model = ContentType.objects.get_for_id(ct_value).model_class()
                        yield (cls._get_base_content_type(model).id, str(fk_value), field.name, field.name)
                    continue
                if isinstance(field, GenericRel):
                    continue
                value = field.value_from_object(object)
                if value is not None:
                    yield (cls._get_base_content_type(field.related_model).id, str(value), field.name, field.name)
            if hasattr(field, 'extract_references'):
                value = field.value_from_object(object)
                if value is not None:
                    yield from ((cls._get_base_content_type(to_model).id, to_object_id, f'{field.name}.{model_path}', f'{field.name}.{content_path}') for (to_model, to_object_id, model_path, content_path) in field.extract_references(value))
        if isinstance(object, ClusterableModel):
            for child_relation in get_all_child_relations(object):
                relation_name = child_relation.get_accessor_name()
                child_objects = getattr(object, relation_name).all()
                for child_object in child_objects:
                    yield from ((to_content_type_id, to_object_id, f'{relation_name}.item.{model_path}', f'{relation_name}.{str(child_object.id)}.{content_path}') for (to_content_type_id, to_object_id, model_path, content_path) in cls._extract_references_from_object(child_object))

    @classmethod
    def _get_content_path_hash(cls, content_path):
        if False:
            while True:
                i = 10
        '\n        Returns a UUID for the given content path. Used to enforce uniqueness.\n\n        Note: MySQL has a limit on the length of fields that are used in unique keys so\n              we need a separate hash field to allow us to support long content paths.\n\n        Args:\n            content_path (str): The content path to get a hash for\n\n        Returns:\n            A UUID instance containing the hash of the given content path\n        '
        return uuid.uuid5(uuid.UUID('bdc70d8b-e7a2-4c2a-bf43-2a3e3fcbbe86'), content_path)

    @classmethod
    def create_or_update_for_object(cls, object):
        if False:
            i = 10
            return i + 15
        '\n        Creates or updates ReferenceIndex records for the given object.\n\n        This method will extract any outbound references from the given object\n        and insert/update them in the database.\n\n        Note: This method must be called within a `django.db.transaction.atomic()` block.\n\n        Args:\n            object (Model): The model instance to create/update ReferenceIndex records for\n        '
        references = set(cls._extract_references_from_object(object))
        content_types = [ContentType.objects.get_for_model(model_or_object, for_concrete_model=False) for model_or_object in [object] + object._meta.get_parent_list()]
        content_type = content_types[0]
        base_content_type = content_types[-1]
        known_content_type_ids = [ct.id for ct in content_types]
        existing_references = {(to_content_type_id, to_object_id, model_path, content_path): (content_type_id, id) for (id, content_type_id, to_content_type_id, to_object_id, model_path, content_path) in cls.objects.filter(base_content_type=base_content_type, object_id=object.pk).values_list('id', 'content_type_id', 'to_content_type', 'to_object_id', 'model_path', 'content_path')}
        new_references = references - set(existing_references.keys())
        bulk_create_kwargs = {}
        if connection.features.supports_ignore_conflicts:
            bulk_create_kwargs['ignore_conflicts'] = True
        cls.objects.bulk_create([cls(content_type=content_type, base_content_type=base_content_type, object_id=object.pk, to_content_type_id=to_content_type_id, to_object_id=to_object_id, model_path=model_path, content_path=content_path, content_path_hash=cls._get_content_path_hash(content_path)) for (to_content_type_id, to_object_id, model_path, content_path) in new_references], **bulk_create_kwargs)
        deleted_reference_ids = []
        for (reference_data, (content_type_id, id)) in existing_references.items():
            if reference_data in references:
                continue
            if content_type_id not in known_content_type_ids:
                continue
            deleted_reference_ids.append(id)
        cls.objects.filter(id__in=deleted_reference_ids).delete()

    @classmethod
    def remove_for_object(cls, object):
        if False:
            i = 10
            return i + 15
        '\n        Deletes all outbound references for the given object.\n\n        Use this before deleting the object itself.\n\n        Args:\n            object (Model): The model instance to delete ReferenceIndex records for\n        '
        base_content_type = cls._get_base_content_type(object)
        cls.objects.filter(base_content_type=base_content_type, object_id=object.pk).delete()

    @classmethod
    def get_references_for_object(cls, object):
        if False:
            while True:
                i = 10
        '\n        Returns all outbound references for the given object.\n\n        Args:\n            object (Model): The model instance to fetch ReferenceIndex records for\n\n        Returns:\n            A QuerySet of ReferenceIndex records\n        '
        return cls.objects.filter(base_content_type_id=cls._get_base_content_type(object), object_id=object.pk)

    @classmethod
    def get_references_to(cls, object):
        if False:
            print('Hello World!')
        '\n        Returns all inbound references for the given object.\n\n        Args:\n            object (Model): The model instance to fetch ReferenceIndex records for\n\n        Returns:\n            A QuerySet of ReferenceIndex records\n        '
        return cls.objects.filter(to_content_type_id=cls._get_base_content_type(object), to_object_id=object.pk)

    @classmethod
    def get_grouped_references_to(cls, object):
        if False:
            return 10
        '\n        Returns all inbound references for the given object, grouped by the object\n        they are found on.\n\n        Args:\n            object (Model): The model instance to fetch ReferenceIndex records for\n\n        Returns:\n            A ReferenceGroups object\n        '
        return cls.get_references_to(object).group_by_source_object()

    @property
    def _content_type(self):
        if False:
            print('Hello World!')
        return ContentType.objects.get_for_id(self.content_type_id)

    @cached_property
    def model_name(self):
        if False:
            while True:
                i = 10
        '\n        The model name of the object from which the reference was extracted.\n        For most cases, this is also where the reference exists on the database\n        (i.e. ``related_field_model_name``). However, for ClusterableModels, the\n        reference is extracted from the parent model.\n\n        Example:\n        A relationship between a BlogPage, BlogPageGalleryImage, and Image\n        is extracted from the BlogPage model, but the reference is stored on\n        on the BlogPageGalleryImage model.\n        '
        return self._content_type.name

    @cached_property
    def related_field_model_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The model name where the reference exists on the database.\n        '
        return self.related_field.model._meta.verbose_name

    @cached_property
    def on_delete(self):
        if False:
            print('Hello World!')
        try:
            return self.reverse_related_field.on_delete
        except AttributeError:
            return models.SET_NULL

    @cached_property
    def source_field(self):
        if False:
            return 10
        '\n        The field from which the reference was extracted.\n        This may be a related field (e.g. ForeignKey), a reverse related field\n        (e.g. ManyToOneRel), a StreamField, or any other field that defines\n        extract_references().\n        '
        model_path_components = self.model_path.split('.')
        field_name = model_path_components[0]
        field = self._content_type.model_class()._meta.get_field(field_name)
        return field

    @cached_property
    def related_field(self):
        if False:
            while True:
                i = 10
        if isinstance(self.source_field, models.ForeignObjectRel):
            return self.source_field.remote_field
        return self.source_field

    @cached_property
    def reverse_related_field(self):
        if False:
            return 10
        return self.related_field.remote_field

    def describe_source_field(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a string describing the field that this reference was extracted from.\n\n        For StreamField, this returns the label of the block that contains the reference.\n        For other fields, this returns the verbose name of the field.\n        '
        field = self.source_field
        model_path_components = self.model_path.split('.')
        if isinstance(field, models.ManyToOneRel):
            child_field = field.related_model._meta.get_field(model_path_components[2])
            return capfirst(child_field.verbose_name)
        elif isinstance(field, StreamField):
            label = f'{capfirst(field.verbose_name)}'
            block = field.stream_block
            block_idx = 1
            while isinstance(block, StreamBlock):
                block = block.child_blocks[model_path_components[block_idx]]
                block_label = capfirst(block.label)
                label += f' → {block_label}'
                block_idx += 1
            return label
        else:
            try:
                field_name = field.verbose_name
            except AttributeError:
                field_name = field.name.replace('_', ' ')
            return capfirst(field_name)

    def describe_on_delete(self):
        if False:
            print('Hello World!')
        '\n        Returns a string describing the action that will be taken when the referenced object is deleted.\n        '
        if self.on_delete == models.CASCADE:
            return _('the %(model_name)s will also be deleted') % {'model_name': self.related_field_model_name}
        if self.on_delete == models.PROTECT:
            return _('prevents deletion')
        if self.on_delete == models.SET_DEFAULT:
            return _('will be set to the default %(model_name)s') % {'model_name': self.related_field_model_name}
        if self.on_delete == models.DO_NOTHING:
            return _('will do nothing')
        if self.on_delete == models.RESTRICT:
            return _('may prevent deletion')
        if hasattr(self.on_delete, 'deconstruct') and self.on_delete.deconstruct()[0] == 'django.db.models.SET':
            return _('will be set to a %(model_name)s specified by the system') % {'model_name': self.related_field_model_name}
        return _('will unset the reference')
ItemBase.wagtail_reference_index_ignore = True