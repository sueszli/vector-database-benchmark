from robyn import Request

def SyncView():
    if False:
        for i in range(10):
            print('nop')

    def get():
        if False:
            return 10
        return 'Hello, world!'

    def post(request: Request):
        if False:
            while True:
                i = 10
        return request.body