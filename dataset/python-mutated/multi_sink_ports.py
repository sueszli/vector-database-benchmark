def source():
    if False:
        i = 10
        return i + 15
    pass

def sinkA(x):
    if False:
        print('Hello World!')
    pass

def sinkB(x):
    if False:
        return 10
    pass

def sinkC(x):
    if False:
        while True:
            i = 10
    pass

def sinkD(x):
    if False:
        while True:
            i = 10
    pass

def split(x):
    if False:
        i = 10
        return i + 15
    y = x._params
    sinkB(y)
    sinkC(y)
    sinkD(y)
    return x

def wrapper(x):
    if False:
        for i in range(10):
            print('nop')
    y = split(x)
    sinkA(y)

def issue():
    if False:
        i = 10
        return i + 15
    x = source()
    wrapper(x)

def splitwrapper(x):
    if False:
        while True:
            i = 10
    return split(x)

class QueryBase:

    def send(self):
        if False:
            while True:
                i = 10
        pass

class Query(QueryBase):
    _params = None

    def send(self):
        if False:
            for i in range(10):
                print('nop')
        return splitwrapper(self)

    def params(self, data):
        if False:
            print('Hello World!')
        self._params = data
        return self

def log_call(params, response):
    if False:
        i = 10
        return i + 15
    sinkA(params)
    sinkA(response)

def wrapper2(x: Query):
    if False:
        for i in range(10):
            print('nop')
    params = x._params
    response = None
    try:
        response = x.send()
    except Exception as ex:
        raise ex
    log_call(params, response)

def issue2():
    if False:
        while True:
            i = 10
    taint = source()
    query = Query().params(taint)
    wrapper2(query)