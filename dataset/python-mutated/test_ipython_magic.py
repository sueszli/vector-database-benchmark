from test.fake_time_util import fake_time
import pytest
cell_code = '\nimport time\n\ndef function_a():\n    function_b()\n    function_c()\n\ndef function_b():\n    function_d()\n\ndef function_c():\n    function_d()\n\ndef function_d():\n    function_e()\n\ndef function_e():\n    time.sleep(0.1)\n\nfunction_a()\n'

@pytest.mark.ipythonmagic
def test_magics(ip):
    if False:
        return 10
    from IPython.utils.capture import capture_output as capture_ipython_output
    with fake_time():
        with capture_ipython_output() as captured:
            ip.run_cell_magic('pyinstrument', line='', cell=cell_code)
    assert len(captured.outputs) == 1
    output = captured.outputs[0]
    assert 'text/html' in output.data
    assert 'text/plain' in output.data
    assert 'function_a' in output.data['text/html']
    assert '<iframe' in output.data['text/html']
    assert 'function_a' in output.data['text/plain']
    assert '- 0.200 function_a' in output.data['text/plain']
    assert '- 0.100 FakeClock.sleep' in output.data['text/plain']
    with fake_time():
        with capture_ipython_output() as captured:
            ip.run_line_magic('pyinstrument', line='function_a()')
    assert len(captured.outputs) == 1
    output = captured.outputs[0]
    assert 'function_a' in output.data['text/plain']
    assert '- 0.100 FakeClock.sleep' in output.data['text/plain']

@pytest.mark.ipythonmagic
def test_magic_empty_line(ip):
    if False:
        return 10
    ip.run_line_magic('pyinstrument', line='')

@pytest.mark.ipythonmagic
def test_magic_no_variable_expansion(ip, capsys):
    if False:
        return 10
    ip.run_line_magic('pyinstrument', line='print("hello {len(\'world\')}")')
    captured = capsys.readouterr()
    assert "hello {len('world')}" in captured.out
    assert 'hello 5' not in captured.out

@pytest.fixture(scope='module')
def session_ip():
    if False:
        return 10
    from IPython.testing.globalipapp import start_ipython
    yield start_ipython()

@pytest.fixture(scope='function')
def ip(session_ip):
    if False:
        i = 10
        return i + 15
    session_ip.run_line_magic(magic_name='load_ext', line='pyinstrument')
    yield session_ip
    session_ip.run_line_magic(magic_name='reset', line='-f')