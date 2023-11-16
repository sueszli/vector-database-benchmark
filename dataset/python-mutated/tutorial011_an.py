from fastapi import Depends, FastAPI
from typing_extensions import Annotated
app = FastAPI()

class FixedContentQueryChecker:

    def __init__(self, fixed_content: str):
        if False:
            i = 10
            return i + 15
        self.fixed_content = fixed_content

    def __call__(self, q: str=''):
        if False:
            print('Hello World!')
        if q:
            return self.fixed_content in q
        return False
checker = FixedContentQueryChecker('bar')

@app.get('/query-checker/')
async def read_query_check(fixed_content_included: Annotated[bool, Depends(checker)]):
    return {'fixed_content_in_query': fixed_content_included}