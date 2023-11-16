from hypothesis import strategies as st

def test_labels_are_cached():
    if False:
        return 10
    x = st.integers()
    assert x.label is x.label

def test_labels_are_distinct():
    if False:
        for i in range(10):
            print('nop')
    assert st.integers().label != st.text().label

@st.composite
def foo(draw):
    if False:
        print('Hello World!')
    return draw(st.none())

@st.composite
def bar(draw):
    if False:
        return 10
    return draw(st.none())

def test_different_composites_have_different_labels():
    if False:
        for i in range(10):
            print('nop')
    assert foo().label != bar().label

def test_one_of_label_is_distinct():
    if False:
        return 10
    a = st.integers()
    b = st.booleans()
    assert st.one_of(a, b).label != st.one_of(b, a).label

def test_lists_label_by_element():
    if False:
        print('Hello World!')
    assert st.lists(st.integers()).label != st.lists(st.booleans()).label

def test_label_of_deferred_strategy_is_well_defined():
    if False:
        print('Hello World!')
    recursive = st.deferred(lambda : st.lists(recursive))
    recursive.label