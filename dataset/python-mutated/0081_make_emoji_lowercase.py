import django.core.validators
from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def emoji_to_lowercase(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    RealmEmoji = apps.get_model('zerver', 'RealmEmoji')
    emoji = RealmEmoji.objects.all()
    for e in emoji:
        e.name = e.name.lower()
        e.save()

class Migration(migrations.Migration):
    dependencies = [('zerver', '0080_realm_description_length')]
    operations = [migrations.RunPython(emoji_to_lowercase, elidable=True), migrations.AlterField(model_name='realmemoji', name='name', field=models.TextField(validators=[django.core.validators.MinLengthValidator(1), django.core.validators.RegexValidator(message='Invalid characters in emoji name', regex='^[0-9a-z.\\-_]+(?<![.\\-_])$')]))]