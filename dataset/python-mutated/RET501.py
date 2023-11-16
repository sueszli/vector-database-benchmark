def x(y):
    if False:
        print('Hello World!')
    if not y:
        return
    return None

class BaseCache:

    def get(self, key: str) -> str | None:
        if False:
            i = 10
            return i + 15
        print(f'{key} not found')
        return None

    def get(self, key: str) -> None:
        if False:
            i = 10
            return i + 15
        print(f'{key} not found')
        return None