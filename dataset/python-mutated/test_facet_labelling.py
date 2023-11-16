from plotnine import aes, as_labeller, facet_grid, facet_wrap, geom_point, ggplot, labeller
from plotnine.data import mtcars

def number_to_word(n):
    if False:
        i = 10
        return i + 15
    lst = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    try:
        return lst[int(n)]
    except IndexError:
        return str(n)
labeller_cols_both = labeller(rows='label_value', cols='label_both', multi_line=False)
labeller_towords = labeller(rows='label_both', cols='label_both', multi_line=False, am=number_to_word, gear=number_to_word)
g = ggplot(mtcars, aes(x='wt', y='mpg')) + geom_point()

def test_label_value():
    if False:
        for i in range(10):
            print('nop')
    p = g + facet_wrap('~ gear', labeller='label_value')
    assert p == 'label_value'

def test_label_both():
    if False:
        i = 10
        return i + 15
    p = g + facet_wrap('~ gear', labeller='label_both')
    assert p == 'label_both'

def test_label_context():
    if False:
        for i in range(10):
            print('nop')
    p = g + facet_wrap('~ gear', labeller='label_context')
    assert p == 'label_context'

def test_label_context_wrap2vars():
    if False:
        print('Hello World!')
    p = g + facet_wrap('~ gear + am', labeller='label_context')
    assert p == 'label_context_wrap2vars'

def test_labeller_cols_both_wrap():
    if False:
        i = 10
        return i + 15
    p = g + facet_wrap('~ gear + am', labeller=labeller_cols_both)
    assert p == 'labeller_cols_both_wrap'

def test_labeller_cols_both_grid():
    if False:
        i = 10
        return i + 15
    p = g + facet_grid('gear ~ am', labeller=labeller_cols_both)
    assert p == 'labeller_cols_both_grid'

def test_labeller_towords():
    if False:
        i = 10
        return i + 15
    p = g + facet_grid('gear ~ am', labeller=labeller_towords)
    assert p == 'labeller_towords'

def test_aslabeller_func_hashtag():
    if False:
        for i in range(10):
            print('nop')
    func = as_labeller(lambda s: f'#{s}')
    p = g + facet_wrap('~ gear + am', labeller=func)
    assert p == 'aslabeller_func_hashtagit'

def test_aslabeller_dict_0tag():
    if False:
        while True:
            i = 10
    func = as_labeller({'0': '<tag>0</tag>'})
    p = g + facet_wrap('~ gear + am', labeller=func)
    assert p == 'aslabeller_dict_0tag'

def test_uneven_num_of_lines():
    if False:
        for i in range(10):
            print('nop')

    @as_labeller
    def func(s):
        if False:
            i = 10
            return i + 15
        if s == '3':
            s = f'{s}\nline2\nline3\nline4'
        return s
    p = g + facet_wrap('~ gear + am', labeller=func)
    assert p == 'uneven_num_of_lines'