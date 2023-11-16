"""
Topic: 字典的集合操作
Desc : 
"""

def dict_commonality():
    if False:
        while True:
            i = 10
    a = {'x': 1, 'y': 2, 'z': 3}
    b = {'w': 10, 'x': 11, 'y': 2}
    print(a.keys() & b.keys())
    print(a.keys() - b.keys())
    print(a.items() & b.items())
    print(type(a.items()))
    for (a, b) in a.items():
        print(a, b)
if __name__ == '__main__':
    dict_commonality()