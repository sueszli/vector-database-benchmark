from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Optional[str] = None
app = FastAPI()

@app.post('/items/')
def create_item(item: Item):
    if False:
        print('Hello World!')
    return item

@app.get('/items/')
def read_items() -> list[Item]:
    if False:
        return 10
    return [Item(name='Portal Gun', description='Device to travel through the multi-rick-verse'), Item(name='Plumbus')]