from e2b import Sandbox

def test_sudo():
    if False:
        while True:
            i = 10
    sandbox = Sandbox()
    process = sandbox.process.start('sudo echo test')
    process.wait()
    output = process.stdout
    assert output == 'test'
    sandbox.close()