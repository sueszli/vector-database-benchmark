from django.db import migrations
from django.conf import settings

def remove_mongo_types(apps, schema_editor):
    if False:
        print('Hello World!')
    db = settings.MONGODB.newsblur_dev
    collections = db.collection_names()
    for collection_name in collections:
        collection = db[collection_name]
        print(' ---> %s...' % collection_name)
        if 'system' in collection_name:
            continue
        collection.update({}, {'$unset': {'_types': 1}}, multi=True)
        index_information = collection.index_information()
        indexes_to_drop = [key for (key, value) in index_information.items() if 'types' in value]
        for index in indexes_to_drop:
            print(' ---> Dropping mongo index %s on %s...' % (index, collection_name))
            collection.drop_index(index)

class Migration(migrations.Migration):
    dependencies = [('rss_feeds', '0001_initial')]
    operations = [migrations.RunPython(remove_mongo_types, migrations.RunPython.noop)]