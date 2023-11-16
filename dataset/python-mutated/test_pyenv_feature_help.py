def test_help(pyenv):
    if False:
        for i in range(10):
            print('nop')
    (stdout, stderr) = pyenv.help()
    stdout = '\r\n'.join(stdout.splitlines()[:2])
    assert (stdout.strip(), stderr) == ('Usage: pyenv <command> [<args>]', '')