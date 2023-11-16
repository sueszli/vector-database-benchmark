class CachedDOData:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def destroy(self):
        if False:
            print('Hello World!')
        pass

    def flush(self):
        if False:
            print('Hello World!')
        pass

    def __getattribute__(self, name: str):
        if False:
            i = 10
            return i + 15
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value) -> None:
        if False:
            return 10
        object.__setattr__(self, name, value)