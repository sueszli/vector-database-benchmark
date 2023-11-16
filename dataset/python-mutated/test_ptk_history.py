import pytest
try:
    import prompt_toolkit
except ImportError:
    pytest.mark.skip(msg='prompt_toolkit is not available')

@pytest.fixture
def history_obj():
    if False:
        while True:
            i = 10
    'Instantiate `PromptToolkitHistory` and append a line string'
    from xonsh.ptk_shell.history import PromptToolkitHistory
    hist = PromptToolkitHistory(load_prev=False)
    hist.append_string('line10')
    return hist

def test_obj(history_obj):
    if False:
        print('Hello World!')
    assert ['line10'] == history_obj.get_strings()
    assert len(history_obj) == 1
    assert ['line10'] == [x for x in history_obj]