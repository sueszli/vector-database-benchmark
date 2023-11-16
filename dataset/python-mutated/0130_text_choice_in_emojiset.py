from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def change_emojiset(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    UserProfile = apps.get_model('zerver', 'UserProfile')
    for user in UserProfile.objects.filter(emoji_alt_code=True):
        user.emojiset = 'text'
        user.save(update_fields=['emojiset'])

def reverse_change_emojiset(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        for i in range(10):
            print('nop')
    UserProfile = apps.get_model('zerver', 'UserProfile')
    for user in UserProfile.objects.filter(emojiset='text'):
        user.emoji_alt_code = True
        user.emojiset = 'google'
        user.save(update_fields=['emoji_alt_code', 'emojiset'])

class Migration(migrations.Migration):
    dependencies = [('zerver', '0129_remove_userprofile_autoscroll_forever')]
    operations = [migrations.AlterField(model_name='userprofile', name='emojiset', field=models.CharField(choices=[('google', 'Google'), ('apple', 'Apple'), ('twitter', 'Twitter'), ('emojione', 'EmojiOne'), ('text', 'Plain text')], default='google', max_length=20)), migrations.RunPython(change_emojiset, reverse_change_emojiset, elidable=True), migrations.RemoveField(model_name='userprofile', name='emoji_alt_code')]