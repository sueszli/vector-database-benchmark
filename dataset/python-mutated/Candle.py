import peewee
from jesse.services.db import database
if database.is_closed():
    database.open_connection()

class Candle(peewee.Model):
    id = peewee.UUIDField(primary_key=True)
    timestamp = peewee.BigIntegerField()
    open = peewee.FloatField()
    close = peewee.FloatField()
    high = peewee.FloatField()
    low = peewee.FloatField()
    volume = peewee.FloatField()
    exchange = peewee.CharField()
    symbol = peewee.CharField()
    timeframe = peewee.CharField()
    is_partial = True

    class Meta:
        from jesse.services.db import database
        database = database.db
        indexes = ((('exchange', 'symbol', 'timeframe', 'timestamp'), True),)

    def __init__(self, attributes: dict=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        peewee.Model.__init__(self, attributes=attributes, **kwargs)
        if attributes is None:
            attributes = {}
        for (a, value) in attributes.items():
            setattr(self, a, value)
if database.is_open():
    Candle.create_table()