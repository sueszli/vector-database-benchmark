from e2b import Sandbox

def test_homedir():
    if False:
        for i in range(10):
            print('nop')
    sandbox = Sandbox()
    process = sandbox.process.start('echo $HOME')
    process.wait()
    output = process.stdout
    assert output == '/home/user'
    sandbox.close()