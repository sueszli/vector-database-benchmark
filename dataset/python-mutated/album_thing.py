from django.db import connection, models
from api.models.photo import Photo
from api.models.user import User, get_deleted_user

class AlbumThing(models.Model):
    title = models.CharField(max_length=512, db_index=True)
    photos = models.ManyToManyField(Photo)
    thing_type = models.CharField(max_length=512, db_index=True, null=True)
    favorited = models.BooleanField(default=False, db_index=True)
    owner = models.ForeignKey(User, on_delete=models.SET(get_deleted_user), default=None)
    shared_to = models.ManyToManyField(User, related_name='album_thing_shared_to')

    class Meta:
        constraints = [models.UniqueConstraint(fields=['title', 'thing_type', 'owner'], name='unique AlbumThing')]

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%d: %s' % (self.id, self.title)

def get_album_thing(title, owner):
    if False:
        print('Hello World!')
    return AlbumThing.objects.get_or_create(title=title, owner=owner)[0]
view_api_album_thing_sql = "\n    api_albumthing_sql as (\n       select title, 'places365_attribute' thing_type, false favorited, owner_id\n       from (select owner_id, jsonb_array_elements_text(jsonb_extract_path(captions_json,  'places365', 'attributes')) title from api_photo ) photo_attribut\n       group by title, thing_type, favorited, owner_id\n       union all\n       select title, 'places365_category' thing_type, false favorited, owner_id\n       from (select owner_id, jsonb_array_elements_text(jsonb_extract_path(captions_json,  'places365', 'categories')) title from api_photo ) photo_attribut\n       group by title, thing_type, favorited, owner_id\n    )"
view_api_album_thing_photos_sql = "\n    api_albumthing_photos_sql as (\n       select api_albumthing.id albumthing_id, photo_id\n       from (select owner_id, jsonb_array_elements_text(jsonb_extract_path(captions_json,  'places365', 'attributes')) title, image_hash photo_id, 'places365_attribute' thing_type from api_photo ) photo_attribut\n       join api_albumthing using (title,thing_type, owner_id )\n       group by api_albumthing.id, photo_id\n       union all\n       select api_albumthing.id albumthing_id, photo_id\n       from (select owner_id, jsonb_array_elements_text(jsonb_extract_path(captions_json,  'places365', 'categories')) title, image_hash photo_id, 'places365_category' thing_type from api_photo ) photo_attribut\n       join api_albumthing using (title,thing_type, owner_id )\n       group by api_albumthing.id, photo_id\n    )\n"

def create_new_album_thing(cursor):
    if False:
        i = 10
        return i + 15
    'This function create albums from all detected thing on photos'
    SQL = '\n        with {}\n        insert into api_albumthing (title, thing_type,favorited, owner_id)\n        select api_albumthing_sql.*\n        from api_albumthing_sql\n        left join api_albumthing using (title, thing_type, owner_id)\n        where  api_albumthing is null;\n    '.replace('{}', view_api_album_thing_sql)
    cursor.execute(SQL)

def create_new_album_thing_photo(cursor):
    if False:
        return 10
    'This function create link between albums thing and photo from all detected thing on photos'
    SQL = '\n        with {}\n        insert into api_albumthing_photos (albumthing_id, photo_id)\n        select api_albumthing_photos_sql.*\n        from api_albumthing_photos_sql\n        left join api_albumthing_photos using (albumthing_id, photo_id)\n        where  api_albumthing_photos is null;\n    '.replace('{}', view_api_album_thing_photos_sql)
    cursor.execute(SQL)

def delete_album_thing_photo(cursor):
    if False:
        return 10
    'This function delete photos form albums thing where thing disappears'
    SQL = '\n        with {}\n        delete\n        from api_albumthing_photos as p\n        where not exists (\n            select 1\n            from api_albumthing_photos_sql\n            where albumthing_id = p.albumthing_id\n                and photo_id = p.photo_id\n            limit 1\n        )\n    '.replace('{}', view_api_album_thing_photos_sql)
    cursor.execute(SQL)

def delete_album_thing(cursor):
    if False:
        print('Hello World!')
    'This function delete albums thing without photos'
    SQL = '\n        with {}\n        delete from api_albumthing\n        where (title, thing_type, owner_id) not in ( select title, thing_type, owner_id from api_albumthing_sql );\n    '.replace('{}', view_api_album_thing_sql)
    cursor.execute(SQL)

def update():
    if False:
        return 10
    with connection.cursor() as cursor:
        create_new_album_thing(cursor)
        create_new_album_thing_photo(cursor)
        delete_album_thing_photo(cursor)
        delete_album_thing(cursor)