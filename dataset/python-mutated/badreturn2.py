from __future__ import annotations

class MyPlugin:
    pass

def plugin(version: str) -> type[MyPlugin]:
    if False:
        for i in range(10):
            print('nop')
    return MyPlugin