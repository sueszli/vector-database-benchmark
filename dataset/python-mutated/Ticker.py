import peewee

class Ticker(peewee.Model):
    id = peewee.UUIDField(primary_key=True)
    timestamp = peewee.BigIntegerField()
    last_price = peewee.FloatField()
    volume = peewee.FloatField()
    high_price = peewee.FloatField()
    low_price = peewee.FloatField()
    symbol = peewee.CharField()
    exchange = peewee.CharField()

    class Meta:
        from jesse.services.db import database
        database = database.db
        indexes = ((('exchange', 'symbol', 'timestamp'), True),)

    def __init__(self, attributes: dict=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        peewee.Model.__init__(self, attributes=attributes, **kwargs)
        if attributes is None:
            attributes = {}
        for (a, value) in attributes.items():
            setattr(self, a, value)