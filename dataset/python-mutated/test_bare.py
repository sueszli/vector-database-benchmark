import pytest
from reflex.components.base.bare import Bare

@pytest.mark.parametrize('contents,expected', [('hello', 'hello'), ('{}', '{}'), ('${default_state.name}', '${default_state.name}'), ('{state.name}', '{state.name}')])
def test_fstrings(contents, expected):
    if False:
        while True:
            i = 10
    'Test that fstrings are rendered correctly.\n\n    Args:\n        contents: The contents of the component.\n        expected: The expected output.\n    '
    comp = Bare.create(contents).render()
    assert comp['contents'] == expected