import pytest
from xonsh.completers.tools import RichCompletion
from xonsh.readline_shell import _render_completions

@pytest.mark.parametrize('prefix, completion, prefix_len, readline_completion', [('', 'a', 0, 'a'), ('a', 'b', 0, 'ab'), ('a', 'b', 1, 'b'), ('adc', 'bc', 2, 'abc'), ('', RichCompletion('x', 0), 0, 'x'), ('', RichCompletion('x', 0, 'aaa', 'aaa'), 0, 'x'), ('a', RichCompletion('b', 1), 0, 'b'), ('a', RichCompletion('b', 0), 1, 'ab'), ('a', RichCompletion('b'), 0, 'ab'), ('a', RichCompletion('b'), 1, 'b')])
def test_render_completions(prefix, completion, prefix_len, readline_completion):
    if False:
        for i in range(10):
            print('nop')
    assert _render_completions({completion}, prefix, prefix_len) == [readline_completion]

@pytest.mark.parametrize('line, exp', [[repr('hello'), 'hello'], ['2 * 3', '6']])
def test_rl_prompt_cmdloop(line, exp, readline_shell, capsys):
    if False:
        while True:
            i = 10
    shell = readline_shell
    shell.use_rawinput = False
    shell.stdin.write(f'{line}\nexit\n')
    shell.stdin.seek(0)
    shell.cmdloop()
    (out, err) = capsys.readouterr()
    assert exp in out.strip()