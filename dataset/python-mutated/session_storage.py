from typing import Any, Dict, List

class SessionStorage:

    def __init__(self, page):
        if False:
            return 10
        self.__page = page
        self.__store: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        if False:
            return 10
        self.__store[key] = value

    def get(self, key: str):
        if False:
            i = 10
            return i + 15
        return self.__store.get(key)

    def contains_key(self, key: str) -> bool:
        if False:
            i = 10
            return i + 15
        return key in self.__store

    def remove(self, key: str):
        if False:
            for i in range(10):
                print('nop')
        self.__store.pop(key)

    def get_keys(self) -> List[str]:
        if False:
            while True:
                i = 10
        return list(self.__store.keys())

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.__store.clear()