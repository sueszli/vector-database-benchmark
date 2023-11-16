from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def fix_bot_type(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    UserProfile = apps.get_model('zerver', 'UserProfile')
    bots = UserProfile.objects.filter(is_bot=True, bot_type=None)
    for bot in bots:
        bot.bot_type = 1
        bot.save()

class Migration(migrations.Migration):
    dependencies = [('zerver', '0084_realmemoji_deactivated')]
    operations = [migrations.RunPython(fix_bot_type, elidable=True)]