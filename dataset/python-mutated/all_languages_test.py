from __future__ import annotations
from pre_commit.all_languages import languages

def test_python_venv_is_an_alias_to_python():
    if False:
        print('Hello World!')
    assert languages['python_venv'] is languages['python']