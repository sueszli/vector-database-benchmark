import ipam.fields
from utilities.json import CustomFieldJSONEncoder
from django.db import migrations, models
import django.db.models.deletion
import taggit.managers


class Migration(migrations.Migration):

    dependencies = [
        ('tenancy', '0004_extend_tag_support'),
        ('extras', '0064_configrevision'),
        ('ipam', '0052_fhrpgroup'),
    ]

    operations = [
        migrations.CreateModel(
            name='ASN',
            fields=[
                ('created', models.DateField(auto_now_add=True, null=True)),
                ('last_updated', models.DateTimeField(auto_now=True, null=True)),
                ('custom_field_data', models.JSONField(blank=True, default=dict, encoder=CustomFieldJSONEncoder)),
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('asn', ipam.fields.ASNField(unique=True)),
                ('description', models.CharField(blank=True, max_length=200)),
                ('rir', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='asns', to='ipam.rir')),
                ('tags', taggit.managers.TaggableManager(through='extras.TaggedItem', to='extras.Tag')),
                ('tenant', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='asns', to='tenancy.tenant')),
            ],
            options={
                'verbose_name': 'ASN',
                'verbose_name_plural': 'ASNs',
                'ordering': ['asn'],
            },
        ),
    ]
