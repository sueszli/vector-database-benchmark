from fastapi import FastAPI
from pydantic import BaseModel

class User(BaseModel):
    name: str
user = User(name='a')
app = FastAPI()

@app.get('/')
def h(u: User) -> str:
    if False:
        while True:
            i = 10
    return 'a'

def closure():
    if False:
        for i in range(10):
            print('nop')
    app = FastAPI()

    @app.get('/')
    def h(u: User) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'a'
    return app