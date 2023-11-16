from ninja import Schema

class Item(Schema):
    name: str
    description: str = None
    price: float
    quantity: int

@api.post('/items')
def create(request, item: Item):
    if False:
        return 10
    return item