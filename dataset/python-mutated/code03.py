from ninja import Schema

class Item(Schema):
    name: str
    description: str = None
    price: float
    quantity: int

@api.post('/items/{item_id}')
def update(request, item_id: int, item: Item, q: str):
    if False:
        i = 10
        return i + 15
    return {'item_id': item_id, 'item': item.dict(), 'q': q}