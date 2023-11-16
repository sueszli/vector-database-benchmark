from typing import Annotated
from fastapi import Depends, FastAPI
app = FastAPI()

class FixedContentQueryChecker:

    def __init__(self, fixed_content: str):
        if False:
            while True:
                i = 10
        self.fixed_content = fixed_content

    def __call__(self, q: str=''):
        if False:
            while True:
                i = 10
        if q:
            return self.fixed_content in q
        return False
checker = FixedContentQueryChecker('bar')

@app.get('/query-checker/')
async def read_query_check(fixed_content_included: Annotated[bool, Depends(checker)]):
    return {'fixed_content_in_query': fixed_content_included}