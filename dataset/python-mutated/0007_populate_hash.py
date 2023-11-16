from django.db import migrations
from django.db.models import Count, Min
from wagtail.embeds.embeds import get_embed_hash

def migrate_forwards(apps, schema_editor):
    if False:
        while True:
            i = 10
    Embed = apps.get_model('wagtailembeds.Embed')
    batch_size = 1500
    batch = []
    for embed in Embed.objects.all().only('id', 'url', 'max_width').iterator():
        embed.hash = get_embed_hash(embed.url, embed.max_width)
        batch.append(embed)
        if len(batch) == batch_size:
            Embed.objects.bulk_update(batch, ['hash'])
            batch.clear()
    if batch:
        Embed.objects.bulk_update(batch, ['hash'])
    duplicates = Embed.objects.values('hash').annotate(hash_count=Count('id'), min_id=Min('id')).filter(hash_count__gt=1)
    for dup in duplicates:
        Embed.objects.filter(hash=dup['hash']).exclude(id=dup['min_id']).delete()

class Migration(migrations.Migration):
    dependencies = [('wagtailembeds', '0006_add_embed_hash')]
    operations = [migrations.RunPython(migrate_forwards, migrations.RunPython.noop)]