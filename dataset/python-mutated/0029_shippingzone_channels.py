from django.db import migrations, models

def add_relations_between_existing_channels_and_shipping_zones(apps, schema_editor):
    if False:
        print('Hello World!')
    Channel = apps.get_model('channel', 'Channel')
    for channel in Channel.objects.iterator():
        zone_ids = set(channel.shipping_method_listings.values_list('shipping_method__shipping_zone_id', flat=True))
        channel.shipping_zones.set(zone_ids)

class Migration(migrations.Migration):
    dependencies = [('channel', '0001_initial'), ('shipping', '0028_auto_20210308_1135')]
    operations = [migrations.AddField(model_name='shippingzone', name='channels', field=models.ManyToManyField(related_name='shipping_zones', to='channel.Channel')), migrations.RunPython(add_relations_between_existing_channels_and_shipping_zones, migrations.RunPython.noop)]