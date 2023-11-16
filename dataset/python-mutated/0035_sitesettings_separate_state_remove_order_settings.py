from django.db import migrations, models

def update_order_settings_values_to_match_first_channel(apps, schema_editor):
    if False:
        while True:
            i = 10
    SiteSettings = apps.get_model('site', 'SiteSettings')
    Channel = apps.get_model('channel', 'Channel')
    channel = Channel.objects.filter(is_active=True).order_by('slug').first()
    if channel:
        SiteSettings.objects.update(automatically_confirm_all_new_orders=channel.automatically_confirm_all_new_orders, automatically_fulfill_non_shippable_gift_card=channel.automatically_fulfill_non_shippable_gift_card)
    else:
        SiteSettings.objects.update(automatically_confirm_all_new_orders=True, automatically_fulfill_non_shippable_gift_card=True)

class Migration(migrations.Migration):
    dependencies = [('site', '0034_sitesettings_limit_quantity_per_checkout'), ('channel', '0008_update_null_order_settings')]
    operations = [migrations.SeparateDatabaseAndState(database_operations=[migrations.AlterField(model_name='sitesettings', name='automatically_fulfill_non_shippable_gift_card', field=models.BooleanField(null=True, blank=True)), migrations.AlterField(model_name='sitesettings', name='automatically_confirm_all_new_orders', field=models.BooleanField(null=True, blank=True)), migrations.RunPython(migrations.RunPython.noop, update_order_settings_values_to_match_first_channel)], state_operations=[migrations.RemoveField(model_name='sitesettings', name='automatically_fulfill_non_shippable_gift_card'), migrations.RemoveField(model_name='sitesettings', name='automatically_confirm_all_new_orders')])]