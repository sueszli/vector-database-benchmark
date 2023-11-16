import json
import logging
from collections import OrderedDict
from django.db.models import JSONField, F, Q, Subquery, OuterRef
from django.db.models.functions import Cast
from django.db.migrations import RunPython
from django.utils.functional import cached_property
from wagtail.blocks import StreamValue
from wagtail.blocks.migrations import utils
logger = logging.getLogger(__name__)

class MigrateStreamData(RunPython):
    """Subclass of RunPython for streamfield data migration operations"""

    def __init__(self, app_name, model_name, field_name, operations_and_block_paths, revisions_from=None, chunk_size=1024, **kwargs):
        if False:
            while True:
                i = 10
        'MigrateStreamData constructor\n\n        Args:\n            app_name (str): Name of the app.\n            model_name (str): Name of the model.\n            field_name (str): Name of the streamfield.\n            operations_and_block_paths (:obj:`list` of :obj:`tuple` of (:obj:`operation`, :obj:`str`)):\n                List of operations and corresponding block paths to apply.\n            revisions_from (:obj:`datetime`, optional): Only revisions created from this date\n                onwards will be updated. Passing `None` updates all revisions. Defaults to `None`.\n                Note that live and latest revisions will be updated regardless of what value this\n                takes.\n            chunk_size (:obj:`int`, optional): chunk size for queryset.iterator and bulk_update.\n                Defaults to 1024.\n            **kwargs: atomic, elidable, hints for superclass RunPython can be given\n\n        Example:\n            Renaming a block named `field1` to `block1`::\n                MigrateStreamData(\n                    app_name="blog",\n                    model_name="BlogPage",\n                    field_name="content",\n                    operations_and_block_paths=[\n                        (RenameStreamChildrenOperation(old_name="field1", new_name="block1"), ""),\n                    ],\n                    revisions_from=datetime.date(2022, 7, 25)\n                ),\n        '
        self.app_name = app_name
        self.model_name = model_name
        self.field_name = field_name
        self.operations_and_block_paths = operations_and_block_paths
        self.revisions_from = revisions_from
        self.chunk_size = chunk_size
        super().__init__(code=self.migrate_stream_data_forward, reverse_code=lambda *args: None, **kwargs)

    def deconstruct(self):
        if False:
            for i in range(10):
                print('nop')
        (_, args, kwargs) = super().deconstruct()
        kwargs['app_name'] = self.app_name
        kwargs['model_name'] = self.model_name
        kwargs['field_name'] = self.field_name
        kwargs['operations_and_block_paths'] = self.operations_and_block_paths
        kwargs['revisions_from'] = self.revisions_from
        kwargs['chunk_size'] = self.chunk_size
        return (self.__class__.__qualname__, args, kwargs)

    @property
    def migration_name_fragment(self):
        if False:
            print('Hello World!')
        fragments = OrderedDict(((op.operation_name_fragment, None) for (op, _) in self.operations_and_block_paths))
        return '_'.join(fragments.keys())

    def migrate_stream_data_forward(self, apps, schema_editor):
        if False:
            for i in range(10):
                print('nop')
        model = apps.get_model(self.app_name, self.model_name)
        try:
            apps.get_model('wagtailcore', 'Revision')
            revision_query_maker = DefaultRevisionQueryMaker(apps, model, self.revisions_from)
        except LookupError:
            revision_query_maker = Wagtail3RevisionQueryMaker(apps, model, self.revisions_from)
        model_queryset = model.objects.annotate(raw_content=Cast(F(self.field_name), JSONField())).all()
        updated_model_instances_buffer = []
        for instance in model_queryset.iterator(chunk_size=self.chunk_size):
            revision_query_maker.append_instance_data_for_revision_query(instance)
            raw_data = instance.raw_content
            for (operation, block_path_str) in self.operations_and_block_paths:
                try:
                    raw_data = utils.apply_changes_to_raw_data(raw_data=raw_data, block_path_str=block_path_str, operation=operation, streamfield=getattr(model, self.field_name))
                except utils.InvalidBlockDefError as e:
                    raise utils.InvalidBlockDefError(instance=instance) from e
            stream_block = getattr(instance, self.field_name).stream_block
            setattr(instance, self.field_name, StreamValue(stream_block, raw_data, is_lazy=True))
            updated_model_instances_buffer.append(instance)
            if len(updated_model_instances_buffer) == self.chunk_size:
                model.objects.bulk_update(updated_model_instances_buffer, [self.field_name])
                updated_model_instances_buffer = []
        if len(updated_model_instances_buffer) > 0:
            model.objects.bulk_update(updated_model_instances_buffer, [self.field_name])
        if not revision_query_maker.has_revisions:
            return
        revision_queryset = revision_query_maker.get_revision_queryset()
        updated_revisions_buffer = []
        for revision in revision_queryset.iterator(chunk_size=self.chunk_size):
            raw_data = json.loads(revision.content[self.field_name])
            for (operation, block_path_str) in self.operations_and_block_paths:
                try:
                    raw_data = utils.apply_changes_to_raw_data(raw_data=raw_data, block_path_str=block_path_str, operation=operation, streamfield=getattr(model, self.field_name))
                except utils.InvalidBlockDefError as e:
                    if not revision_query_maker.get_is_live_or_latest_revision(revision):
                        logger.exception(utils.InvalidBlockDefError(revision=revision, instance=instance))
                        continue
                    else:
                        raise utils.InvalidBlockDefError(revision=revision, instance=instance) from e
            revision.content[self.field_name] = json.dumps(raw_data)
            updated_revisions_buffer.append(revision)
            if len(updated_revisions_buffer) == self.chunk_size:
                revision_query_maker.bulk_update(updated_revisions_buffer)
                updated_revisions_buffer = []
        if len(updated_revisions_buffer) > 0:
            revision_query_maker.bulk_update(updated_revisions_buffer)

class AbstractRevisionQueryMaker:
    """Helper class for making the revision query needed for the data migration"""

    def __init__(self, apps, model, revisions_from):
        if False:
            return 10
        self.apps = apps
        self.model = model
        self.revisions_from = revisions_from
        self.RevisionModel = self.get_revision_model()
        self.has_revisions = self.get_has_revisions()
        if self.has_revisions:
            self.instance_field_revision_ids = set()

    def get_revision_model(self):
        if False:
            return 10
        raise NotImplementedError

    def get_has_revisions(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def append_instance_data_for_revision_query(self, instance):
        if False:
            return 10
        raise NotImplementedError

    def _make_revision_query(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def get_revision_queryset(self):
        if False:
            i = 10
            return i + 15
        revision_query = self._make_revision_query()
        return self.RevisionModel.objects.filter(revision_query)

    def bulk_update(self, data):
        if False:
            while True:
                i = 10
        self.RevisionModel.objects.bulk_update(data, ['content'])

    def get_is_live_or_latest_revision(self, revision):
        if False:
            return 10
        raise NotImplementedError

class Wagtail3RevisionQueryMaker(AbstractRevisionQueryMaker):
    """Revision Query maker to support Wagtail 3"""

    def __init__(self, apps, model, revisions_from):
        if False:
            print('Hello World!')
        self.page_ids = []
        super().__init__(apps, model, revisions_from)

    def get_revision_model(self):
        if False:
            print('Hello World!')
        return self.apps.get_model('wagtailcore', 'PageRevision')

    def get_has_revisions(self):
        if False:
            print('Hello World!')
        return issubclass(self.model, self.apps.get_model('wagtailcore', 'Page'))

    def append_instance_data_for_revision_query(self, instance):
        if False:
            i = 10
            return i + 15
        if self.has_revisions:
            self.page_ids.append(instance.id)
            self.instance_field_revision_ids.add(instance.live_revision_id)

    def _make_revision_query(self):
        if False:
            while True:
                i = 10
        if self.revisions_from is not None:
            revision_query = Q(created_at__gte=self.revisions_from, page_id__in=self.page_ids)
            revision_query = revision_query | Q(id__in=self.instance_field_revision_ids)
            revision_query = revision_query | Q(id__in=Subquery(self.RevisionModel.objects.filter(page_id=OuterRef('page_id')).order_by('-created_at', '-id').values_list('id', flat=True)[:1]), page_id__in=self.page_ids)
            return revision_query
        else:
            return Q(page_id__in=self.page_ids)

    def get_is_live_or_latest_revision(self, revision):
        if False:
            return 10
        if revision.id in self.instance_field_revision_ids:
            return True
        return revision.id in self._latest_revision_ids

    @cached_property
    def _latest_revision_ids(self):
        if False:
            return 10
        return self.RevisionModel.objects.filter(id__in=Subquery(self.RevisionModel.objects.filter(page_id=OuterRef('page_id')).order_by('-created_at', '-id').values_list('id', flat=True)[:1]), page_id__in=self.page_ids).values_list('id', flat=True)

class DefaultRevisionQueryMaker(AbstractRevisionQueryMaker):
    """Revision Query Maker for Wagtail 4+"""

    def __init__(self, apps, model, revisions_from):
        if False:
            print('Hello World!')
        self.has_live_revisions = False
        self.has_latest_revisions = False
        super().__init__(apps, model, revisions_from)

    def get_revision_model(self):
        if False:
            while True:
                i = 10
        return self.apps.get_model('wagtailcore', 'Revision')

    def get_has_revisions(self):
        if False:
            return 10
        self.has_latest_revisions = hasattr(self.model, 'latest_revision') and self.model.latest_revision.field.remote_field.model is self.RevisionModel
        self.has_live_revisions = hasattr(self.model, 'live_revision') and self.model.live_revision.field.remote_field.model is self.RevisionModel
        return self.has_latest_revisions or self.has_live_revisions

    def append_instance_data_for_revision_query(self, instance):
        if False:
            while True:
                i = 10
        if self.has_revisions:
            if self.has_latest_revisions:
                self.instance_field_revision_ids.add(instance.latest_revision_id)
            if self.has_live_revisions:
                self.instance_field_revision_ids.add(instance.live_revision_id)

    def _make_revision_query(self):
        if False:
            for i in range(10):
                print('nop')
        ContentType = self.apps.get_model('contenttypes', 'ContentType')
        contenttype_id = ContentType.objects.get_for_model(self.model).id
        if self.revisions_from is not None:
            revision_query = Q(created_at__gte=self.revisions_from, content_type_id=contenttype_id)
            revision_query = revision_query | Q(id__in=self.instance_field_revision_ids)
            return revision_query
        else:
            return Q(content_type_id=contenttype_id)

    def get_is_live_or_latest_revision(self, revision):
        if False:
            for i in range(10):
                print('nop')
        return revision.id in self.instance_field_revision_ids