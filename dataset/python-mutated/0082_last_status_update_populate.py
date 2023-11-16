from django.db import migrations
from django.utils import timezone
import logging
logger = logging.getLogger(__name__)

def populate_last_status_update(apps, schema_editor):
    if False:
        while True:
            i = 10
    logger.info('Setting last_status_update timestamp on findings to be initially equal to last_reviewed timestamp (may take a while)')
    now = timezone.now()
    Finding = apps.get_model('dojo', 'Finding')
    findings = Finding.objects.order_by('id').only('id', 'is_Mitigated', 'mitigated')
    page_size = 1000
    total_count = Finding.objects.filter(id__gt=0).count()
    logger.debug('found %d findings to update:', total_count)
    i = 0
    batch = []
    last_id = 0
    total_pages = total_count // page_size + 2
    for p in range(1, total_pages):
        page = findings.filter(id__gt=last_id)[:page_size]
        for find in page:
            i += 1
            last_id = find.id
            if find.is_Mitigated:
                find.last_status_update = find.mitigated
            else:
                find.last_status_update = None
            batch.append(find)
            if i > 0 and i % page_size == 0:
                Finding.objects.bulk_update(batch, ['last_status_update'])
                batch = []
                logger.info('%s out of %s findings processed ...', i, total_count)
    Finding.objects.bulk_update(batch, ['last_status_update'])
    batch = []
    logger.info('%s out of %s findings processed ...', i, total_count)

class Migration(migrations.Migration):
    dependencies = [('dojo', '0081_last_status_update')]
    operations = [migrations.RunPython(populate_last_status_update, migrations.RunPython.noop)]