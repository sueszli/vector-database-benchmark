import common.db.fields
from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion
import simple_history.models
import uuid
from django.utils import timezone
from django.db import migrations, transaction

def migrate_old_authbook_to_history(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    authbook_model = apps.get_model('assets', 'AuthBook')
    history_model = apps.get_model('assets', 'HistoricalAuthBook')
    db_alias = schema_editor.connection.alias
    print()
    while True:
        authbooks = authbook_model.objects.using(db_alias).filter(is_latest=False)[:1000]
        if not authbooks:
            break
        historys = []
        authbook_ids = []
        for authbook in authbooks:
            authbook_ids.append(authbook.id)
            history = history_model()
            for attr in ['id', 'username', 'password', 'private_key', 'public_key', 'version', 'comment', 'created_by', 'asset', 'date_created', 'date_updated']:
                setattr(history, attr, getattr(authbook, attr))
            history.history_type = '-'
            history.history_date = timezone.now()
            historys.append(history)
        with transaction.atomic():
            print('  Migrate old auth book to history table: {} items'.format(len(authbook_ids)))
            history_model.objects.bulk_create(historys, ignore_conflicts=True)
            authbook_model.objects.filter(id__in=authbook_ids).delete()

class Migration(migrations.Migration):
    dependencies = [migrations.swappable_dependency(settings.AUTH_USER_MODEL), ('assets', '0071_systemuser_type')]
    operations = [migrations.CreateModel(name='HistoricalAuthBook', fields=[('org_id', models.CharField(blank=True, db_index=True, default='', max_length=36, verbose_name='Organization')), ('id', models.UUIDField(db_index=True, default=uuid.uuid4)), ('name', models.CharField(max_length=128, verbose_name='Name')), ('username', models.CharField(blank=True, db_index=True, max_length=128, validators=[django.core.validators.RegexValidator('^[0-9a-zA-Z_@\\-\\.]*$', 'Special char not allowed')], verbose_name='Username')), ('password', common.db.fields.EncryptCharField(blank=True, max_length=256, null=True, verbose_name='Password')), ('private_key', common.db.fields.EncryptTextField(blank=True, null=True, verbose_name='SSH private key')), ('public_key', common.db.fields.EncryptTextField(blank=True, null=True, verbose_name='SSH public key')), ('comment', models.TextField(blank=True, verbose_name='Comment')), ('date_created', models.DateTimeField(blank=True, editable=False, verbose_name='Date created')), ('date_updated', models.DateTimeField(blank=True, editable=False, verbose_name='Date updated')), ('created_by', models.CharField(max_length=128, null=True, verbose_name='Created by')), ('version', models.IntegerField(default=1, verbose_name='Version')), ('is_latest', models.BooleanField(default=False, verbose_name='Latest version')), ('history_id', models.AutoField(primary_key=True, serialize=False)), ('history_date', models.DateTimeField()), ('history_change_reason', models.CharField(max_length=100, null=True)), ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)), ('asset', models.ForeignKey(blank=True, db_constraint=False, null=True, on_delete=django.db.models.deletion.DO_NOTHING, related_name='+', to='assets.asset', verbose_name='Asset')), ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL))], options={'verbose_name': 'historical AuthBook', 'ordering': ('-history_date', '-history_id'), 'get_latest_by': 'history_date'}, bases=(simple_history.models.HistoricalChanges, models.Model)), migrations.RunPython(migrate_old_authbook_to_history)]