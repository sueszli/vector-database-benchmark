from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import Count

def fix_duplicate_attachments(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        return 10
    'Migration 0041 had a bug, where if multiple messages referenced the\n    same attachment, rather than creating a single attachment object\n    for all of them, we would incorrectly create one for each message.\n    This results in exceptions looking up the Attachment object\n    corresponding to a file that was used in multiple messages that\n    predate migration 0041.\n\n    This migration fixes this by removing the duplicates, moving their\n    messages onto a single canonical Attachment object (per path_id).\n    '
    Attachment = apps.get_model('zerver', 'Attachment')
    for group in Attachment.objects.values('path_id').annotate(Count('id')).order_by().filter(id__count__gt=1):
        attachments = sorted(Attachment.objects.filter(path_id=group['path_id']).order_by('id'), key=lambda x: min(x.messages.all().values_list('id')[0]))
        surviving = attachments[0]
        to_cleanup = attachments[1:]
        for a in to_cleanup:
            for msg in a.messages.all():
                surviving.messages.add(msg)
            surviving.is_realm_public = surviving.is_realm_public or a.is_realm_public
            surviving.save()
            a.delete()

class Migration(migrations.Migration):
    dependencies = [('zerver', '0073_custom_profile_fields')]
    operations = [migrations.RunPython(fix_duplicate_attachments, elidable=True)]