import ivy
import jax
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.frontends.jax._src.tree_util import tree_leaves, tree_map
import hypothesis.strategies as st

@st.composite
def _tree_dict_strategy(draw):
    if False:
        while True:
            i = 10
    return draw(tree_strategy())

def leaf_strategy():
    if False:
        for i in range(10):
            print('nop')
    return st.lists(st.integers(1, 10)).map(ivy.array)

@handle_frontend_test(fn_tree='jax._src.tree_util.tree_leaves', tree=_tree_dict_strategy())
def test_jax_tree_leaves(*, tree, test_flags, fn_tree, frontend, on_device, backend_fw):
    if False:
        i = 10
        return i + 15
    ivy.set_backend(backend_fw)
    result = tree_leaves(tree)
    expected = jax.tree_util.tree_leaves(tree)
    assert result == expected
    ivy.previous_backend()

@handle_frontend_test(fn_tree='jax._src.tree_util.tree_map', tree=_tree_dict_strategy())
def test_jax_tree_map(*, tree, test_flags, fn_tree, frontend, on_device, backend_fw):
    if False:
        i = 10
        return i + 15
    ivy.set_backend(backend_fw)

    def square(x):
        if False:
            return 10
        if isinstance(x, ivy.Array):
            return ivy.square(x)
        else:
            return x ** 2
    result = tree_map(square, tree)
    expected = ivy.square(ivy.Container(tree))
    assert ivy.equal(ivy.Container(result), expected)
    ivy.previous_backend()

def tree_strategy(max_depth=2):
    if False:
        print('Hello World!')
    if max_depth == 0:
        return leaf_strategy()
    else:
        return st.dictionaries(keys=st.one_of(*[st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=1).filter(lambda x: x not in used_keys) for used_keys in [set()]]), values=st.one_of(leaf_strategy(), tree_strategy(max_depth - 1)), min_size=1, max_size=10)