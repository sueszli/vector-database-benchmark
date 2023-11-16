import pytest
from io import BytesIO
from thefuck.rules.go_unknown_command import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def build_misspelled_output():
    if False:
        for i in range(10):
            print('nop')
    return "go bulid: unknown command\nRun 'go help' for usage."

@pytest.fixture
def go_stderr(mocker):
    if False:
        for i in range(10):
            print('nop')
    stderr = b'Go is a tool for managing Go source code.\n\nUsage:\n\n\tgo <command> [arguments]\n\nThe commands are:\n\n\tbug         start a bug report\n\tbuild       compile packages and dependencies\n\tclean       remove object files and cached files\n\tdoc         show documentation for package or symbol\n\tenv         print Go environment information\n\tfix         update packages to use new APIs\n\tfmt         gofmt (reformat) package sources\n\tgenerate    generate Go files by processing source\n\tget         add dependencies to current module and install them\n\tinstall     compile and install packages and dependencies\n\tlist        list packages or modules\n\tmod         module maintenance\n\trun         compile and run Go program\n\ttest        test packages\n\ttool        run specified go tool\n\tversion     print Go version\n\tvet         report likely mistakes in packages\n\nUse "go help <command>" for more information about a command.\n\nAdditional help topics:\n\n\tbuildconstraint build constraints\n\tbuildmode       build modes\n\tc               calling between Go and C\n\tcache           build and test caching\n\tenvironment     environment variables\n\tfiletype        file types\n\tgo.mod          the go.mod file\n\tgopath          GOPATH environment variable\n\tgopath-get      legacy GOPATH go get\n\tgoproxy         module proxy protocol\n\timportpath      import path syntax\n\tmodules         modules, module versions, and more\n\tmodule-get      module-aware go get\n\tmodule-auth     module authentication using go.sum\n\tmodule-private  module configuration for non-public modules\n\tpackages        package lists and patterns\n\ttestflag        testing flags\n\ttestfunc        testing functions\n\nUse "go help <topic>" for more information about that topic.\n\n'
    mock = mocker.patch('subprocess.Popen')
    mock.return_value.stderr = BytesIO(stderr)
    return mock

def test_match(build_misspelled_output):
    if False:
        print('Hello World!')
    assert match(Command('go bulid', build_misspelled_output))

def test_not_match():
    if False:
        print('Hello World!')
    assert not match(Command('go run', 'go run: no go files listed'))

@pytest.mark.usefixtures('no_memoize', 'go_stderr')
def test_get_new_command(build_misspelled_output):
    if False:
        while True:
            i = 10
    assert get_new_command(Command('go bulid', build_misspelled_output)) == 'go build'