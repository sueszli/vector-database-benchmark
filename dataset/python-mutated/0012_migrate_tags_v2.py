from typing import Any, List, Tuple
from django.core.paginator import Paginator
from django.db import migrations
from django.db.models import Q
from posthog.models.tag import tagify

def forwards(apps, schema_editor):
    if False:
        while True:
            i = 10
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info('ee/0012_migrate_tags_v2_start')
    Tag = apps.get_model('posthog', 'Tag')
    TaggedItem = apps.get_model('posthog', 'TaggedItem')
    EnterpriseEventDefinition = apps.get_model('ee', 'EnterpriseEventDefinition')
    EnterprisePropertyDefinition = apps.get_model('ee', 'EnterprisePropertyDefinition')
    createables: List[Tuple[Any, Any]] = []
    batch_size = 1000
    event_definition_paginator = Paginator(EnterpriseEventDefinition.objects.exclude(Q(deprecated_tags__isnull=True) | Q(deprecated_tags=[])).order_by('created_at').values_list('deprecated_tags', 'team_id', 'id'), batch_size)
    for event_definition_page in event_definition_paginator.page_range:
        logger.info('event_definition_tag_batch_get_start', limit=batch_size, offset=(event_definition_page - 1) * batch_size)
        event_definitions = iter(event_definition_paginator.get_page(event_definition_page))
        for (tags, team_id, event_definition_id) in event_definitions:
            unique_tags = set((tagify(t) for t in tags if isinstance(t, str) and t.strip() != ''))
            for tag in unique_tags:
                temp_tag = Tag(name=tag, team_id=team_id)
                createables.append((temp_tag, TaggedItem(event_definition_id=event_definition_id, tag_id=temp_tag.id)))
    logger.info('event_definition_tag_get_end', tags_count=len(createables))
    num_event_definition_tags = len(createables)
    property_definition_paginator = Paginator(EnterprisePropertyDefinition.objects.exclude(Q(deprecated_tags__isnull=True) | Q(deprecated_tags=[])).order_by('updated_at').values_list('deprecated_tags', 'team_id', 'id'), batch_size)
    for property_definition_page in property_definition_paginator.page_range:
        logger.info('property_definition_tag_batch_get_start', limit=batch_size, offset=(property_definition_page - 1) * batch_size)
        property_definitions = iter(property_definition_paginator.get_page(property_definition_page))
        for (tags, team_id, property_definition_id) in property_definitions:
            unique_tags = set((tagify(t) for t in tags if isinstance(t, str) and t.strip() != ''))
            for tag in unique_tags:
                temp_tag = Tag(name=tag, team_id=team_id)
                createables.append((temp_tag, TaggedItem(property_definition_id=property_definition_id, tag_id=temp_tag.id)))
    logger.info('property_definition_tag_get_end', tags_count=len(createables) - num_event_definition_tags)
    createables = sorted(createables, key=lambda pair: pair[0].name)
    tags_to_create = [tag for (tag, _) in createables]
    Tag.objects.bulk_create(tags_to_create, ignore_conflicts=True, batch_size=batch_size)
    logger.info('tags_bulk_created')
    for offset in range(0, len(tags_to_create), batch_size):
        logger.info('tagged_item_batch_create_start', limit=batch_size, offset=offset)
        batch = tags_to_create[offset:offset + batch_size]
        created_tags = Tag.objects.in_bulk([t.id for t in batch])
        createable_batch = createables[offset:offset + batch_size]
        for (tag, tagged_item) in createable_batch:
            if tag.id in created_tags:
                tagged_item.tag_id = created_tags[tag.id].id
            else:
                tagged_item.tag_id = Tag.objects.filter(name=tag.name, team_id=tag.team_id).first().id
        TaggedItem.objects.bulk_create([tagged_item for (_, tagged_item) in createable_batch], ignore_conflicts=True, batch_size=batch_size)
    logger.info('ee/0012_migrate_tags_v2_end')

def reverse(apps, schema_editor):
    if False:
        print('Hello World!')
    TaggedItem = apps.get_model('posthog', 'TaggedItem')
    TaggedItem.objects.filter(Q(event_definition_id__isnull=False) | Q(property_definition_id__isnull=False)).delete()

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('ee', '0011_add_tags_back'), ('posthog', '0218_uniqueness_constraint_tagged_items')]
    operations = [migrations.RunPython(forwards, reverse)]