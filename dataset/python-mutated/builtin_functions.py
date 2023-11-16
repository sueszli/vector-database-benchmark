from integration_test.taint import source, sink

def test_enumerate():
    if False:
        while True:
            i = 10
    elements = ['a', source(), 'b']
    for (index, value) in enumerate(elements):
        sink(index)
        sink(value)
    elements = [{}, {}, {'a': source()}]
    for (index, value) in enumerate(elements):
        sink(index)
        sink(value)
        sink(value['a'])
        sink(value['b'])

def test_sorted(i: int):
    if False:
        for i in range(10):
            print('nop')
    elements = ['a', source(), 'b']
    elements = sorted(elements)
    sink(elements[0])
    elements = [(0, 'a'), (0, source()), (0, 'b')]
    elements = sorted(elements)
    sink(elements[0][0])
    sink(elements[0][1])
    sink(elements[i][1])
    d = {(0, 0): 'a', (0, source()): 'b'}
    elements = sorted(d)
    sink(elements[i][1])
    sink(elements[i][0])

def test_reversed(i: int):
    if False:
        for i in range(10):
            print('nop')
    elements = ['a', 'b', source()]
    elements = reversed(elements)
    sink(elements[0])
    elements = [(0, 'a'), (0, source())]
    elements = reversed(elements)
    sink(elements[0][0])
    sink(elements[1][0])
    sink(elements[i][0])
    sink(elements[0][1])
    sink(elements[1][1])
    sink(elements[i][1])
    d = {(0, 0): 'a', (0, source()): 'b'}
    elements = reversed(d)
    sink(elements[i][1])
    sink(elements[i][0])

def test_map_lambda(i: int):
    if False:
        i = 10
        return i + 15
    elements = list(map(lambda x: x, [source()]))
    sink(elements[0])
    sink(elements[i])
    elements = list(map(lambda x: x, [0, source(), 0]))
    sink(elements[i])
    sink(elements[1])
    sink(elements[0])
    elements = list(map(lambda x: {'a': x, 'b': 'safe'}, [source()]))
    sink(elements[i])
    sink(elements[i]['a'])
    sink(elements[i]['b'])
    elements = list(map(lambda x: x['a'], [{'a': source(), 'b': 'safe'}]))
    sink(elements[i])
    elements = list(map(lambda x: x['b'], [{'a': source(), 'b': 'safe'}]))
    sink(elements[i])
    elements = list(map(lambda x: source(), ['safe']))
    sink(elements[i])
    elements = list(map(lambda x: sink(x), [source()]))

def test_filter_lambda(i: int):
    if False:
        while True:
            i = 10
    elements = list(filter(lambda x: x != 0, [source()]))
    sink(elements[0])
    sink(elements[i])
    elements = list(filter(lambda x: x != 0, [0, source(), 1]))
    sink(elements[i])
    sink(elements[0])
    sink(elements[1])
    elements = list(filter(lambda x: x['a'], [{'a': source(), 'b': 'safe'}]))
    sink(elements[i])
    sink(elements[i]['a'])
    sink(elements[i]['b'])
    elements = list(filter(lambda x: sink(x), [source()]))
    elements = list(filter(lambda x: x, {source(): 0, 'b': 1}))
    sink(elements[i])
    elements = list(filter(lambda x: x, {(0, source()): 0, 'b': 1}))
    sink(elements[i])
    sink(elements[i][0])
    sink(elements[i][1])

def test_next_iter():
    if False:
        i = 10
        return i + 15
    elements = [source()]
    sink(next(iter(elements)))
    elements = [0, source(), 2]
    i = iter(elements)
    sink(next(i))
    sink(next(i))
    sink(next(i))
    elements = [{'bad': source(), 'good': 'safe'}]
    element = next(iter(elements))
    sink(element['bad'])
    sink(element['good'])
    d = {'a': source()}
    sink(next(iter(d)))
    d = {source(): 0}
    sink(next(iter(d)))
    element = next(iter([]), source())
    sink(element)
    element = next(iter([]), {'bad': source(), 'good': 'safe'})
    sink(element)
    sink(element['bad'])
    sink(element['good'])