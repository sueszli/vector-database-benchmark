import orjson
from django.db import connection, migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from psycopg2.sql import SQL

def convert_muted_topics(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    stream_query = SQL('\n        SELECT\n            zerver_stream.name,\n            zerver_stream.realm_id,\n            zerver_stream.id,\n            zerver_recipient.id\n        FROM\n            zerver_stream\n        INNER JOIN zerver_recipient ON (\n            zerver_recipient.type_id = zerver_stream.id AND\n            zerver_recipient.type = 2\n        )\n    ')
    stream_dict = {}
    with connection.cursor() as cursor:
        cursor.execute(stream_query)
        rows = cursor.fetchall()
        for (stream_name, realm_id, stream_id, recipient_id) in rows:
            stream_name = stream_name.lower()
            stream_dict[stream_name, realm_id] = (stream_id, recipient_id)
    UserProfile = apps.get_model('zerver', 'UserProfile')
    MutedTopic = apps.get_model('zerver', 'MutedTopic')
    new_objs = []
    user_query = UserProfile.objects.values('id', 'realm_id', 'muted_topics')
    for row in user_query:
        user_profile_id = row['id']
        realm_id = row['realm_id']
        muted_topics = row['muted_topics']
        tups = orjson.loads(muted_topics)
        for (stream_name, topic_name) in tups:
            stream_name = stream_name.lower()
            val = stream_dict.get((stream_name, realm_id))
            if val is not None:
                (stream_id, recipient_id) = val
                muted_topic = MutedTopic(user_profile_id=user_profile_id, stream_id=stream_id, recipient_id=recipient_id, topic_name=topic_name)
                new_objs.append(muted_topic)
    with connection.cursor() as cursor:
        cursor.execute('DELETE from zerver_mutedtopic')
    MutedTopic.objects.bulk_create(new_objs)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0101_muted_topic')]
    operations = [migrations.RunPython(convert_muted_topics, elidable=True)]