def some_function():
    if False:
        while True:
            i = 10
    return ['some_string']

def some_other_function():
    if False:
        for i in range(10):
            print('nop')
    (some_variable,) = some_function()
    print(some_variable)

def bug(d):
    if False:
        return 10
    d[1:2, 1:2] += 1
empty_tup = ()
one_item_tup = ('item1',)
one_item_tup_without_parentheses = ('item',)
many_items_tup = ('item1', 'item2', 'item3')