"""
Topic: 合并多个字典或映射
Desc : 
"""
from collections import ChainMap

def combine_map():
    if False:
        i = 10
        return i + 15
    a = {'x': 1, 'z': 3}
    b = {'y': 2, 'z': 4}
    c = ChainMap(a, b)
    print(c['x'])
    print(c['y'])
    print(c['z'])
    print(len(c))
    print(list(c.keys()))
    print(list(c.values()))
    c['z'] = 10
    c['w'] = 40
    del c['x']
    print(a)
    values = ChainMap()
    values['x'] = 1
    values = values.new_child()
    values['x'] = 2
    values = values.new_child()
    values['x'] = 3
    print(values)
    print(values['x'])
    values = values.parents
    print(values['x'])
    values = values.parents
    print(values['x'])
    print(values)
if __name__ == '__main__':
    combine_map()