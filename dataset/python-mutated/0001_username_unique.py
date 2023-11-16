from django.db import migrations
from django.conf import settings
import pymongo

def remove_unique_index(apps, schema_editor):
    if False:
        return 10
    social_profile = sp = settings.MONGODB[settings.MONGO_DB_NAME].social_profile
    try:
        social_profile.drop_index('username_1')
    except pymongo.errors.OperationFailure:
        print(" ***> Couldn't delete username_1 index on social_profile collection. Already deleted?")
        pass

class Migration(migrations.Migration):
    dependencies = []
    operations = [migrations.RunPython(remove_unique_index)]