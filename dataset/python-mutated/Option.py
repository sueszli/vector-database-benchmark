import peewee
from jesse.services.db import database
if database.is_closed():
    database.open_connection()

class Option(peewee.Model):
    id = peewee.UUIDField(primary_key=True)
    updated_at = peewee.BigIntegerField()
    type = peewee.CharField()
    json = peewee.TextField()

    class Meta:
        from jesse.services.db import database
        database = database.db

    def __init__(self, attributes=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        peewee.Model.__init__(self, attributes=attributes, **kwargs)
        if attributes is None:
            attributes = {}
        for a in attributes:
            setattr(self, a, attributes[a])
if database.is_open():
    Option.create_table()