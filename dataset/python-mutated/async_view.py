from robyn import Request

def AsyncView():
    if False:
        while True:
            i = 10

    async def get():
        return 'Hello, world!'

    async def post(request: Request):
        return request.body