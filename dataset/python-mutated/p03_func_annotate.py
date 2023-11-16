"""
Topic: 函数注解元信息
Desc : 
"""

def add(x: int, y: int) -> int:
    if False:
        return 10
    return x + y
help(add)
print(add.__annotations__)