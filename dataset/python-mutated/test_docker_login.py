from thefuck.rules.docker_login import match, get_new_command
from thefuck.types import Command

def test_match():
    if False:
        return 10
    err_response1 = "\n    Sending build context to Docker daemon  118.8kB\nStep 1/6 : FROM foo/bar:fdb7c6d\npull access denied for foo/bar, repository does not exist or may require 'docker login'\n"
    assert match(Command('docker build -t artifactory:9090/foo/bar:fdb7c6d .', err_response1))
    err_response2 = "\n    The push refers to repository [artifactory:9090/foo/bar]\npush access denied for foo/bar, repository does not exist or may require 'docker login'\n"
    assert match(Command('docker push artifactory:9090/foo/bar:fdb7c6d', err_response2))
    err_response3 = '\n    docker push artifactory:9090/foo/bar:fdb7c6d\nThe push refers to repository [artifactory:9090/foo/bar]\n9c29c7ad209d: Preparing\n71f3ad53dfe0: Preparing\nf58ee068224c: Preparing\naeddc924d0f7: Preparing\nc2040e5d6363: Preparing\n4d42df4f350f: Preparing\n35723dab26f9: Preparing\n71f3ad53dfe0: Pushed\ncb95fa0faeb1: Layer already exists\n'
    assert not match(Command('docker push artifactory:9090/foo/bar:fdb7c6d', err_response3))

def test_get_new_command():
    if False:
        print('Hello World!')
    assert get_new_command(Command('docker build -t artifactory:9090/foo/bar:fdb7c6d .', '')) == 'docker login && docker build -t artifactory:9090/foo/bar:fdb7c6d .'
    assert get_new_command(Command('docker push artifactory:9090/foo/bar:fdb7c6d', '')) == 'docker login && docker push artifactory:9090/foo/bar:fdb7c6d'