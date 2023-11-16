from ninja import Schema

class Item(Schema):
    name: str
    description: str = None
    price: float
    quantity: int

@api.put('/items/{item_id}')
def update(request, item_id: int, item: Item):
    if False:
        for i in range(10):
            print('nop')
    return {'item_id': item_id, 'item': item.dict()}