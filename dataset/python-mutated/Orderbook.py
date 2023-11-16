import peewee

class Orderbook(peewee.Model):
    id = peewee.UUIDField(primary_key=True)
    timestamp = peewee.BigIntegerField()
    symbol = peewee.CharField()
    exchange = peewee.CharField()
    data = peewee.BlobField()

    class Meta:
        from jesse.services.db import database
        database = database.db
        indexes = ((('exchange', 'symbol', 'timestamp'), True),)

    def __init__(self, attributes: dict=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        peewee.Model.__init__(self, attributes=attributes, **kwargs)
        if attributes is None:
            attributes = {}
        for (a, value) in attributes.items():
            setattr(self, a, value)