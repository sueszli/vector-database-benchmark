import os
from contextlib import contextmanager

@contextmanager
def inside_dir(dirpath):
    if False:
        print('Hello World!')
    '\n    Execute code from inside the given directory\n    :param dirpath: String, path of the directory the command is being run.\n    '
    old_path = os.getcwd()
    try:
        os.chdir(dirpath)
        yield
    finally:
        os.chdir(old_path)

def test_project_tree(cookies):
    if False:
        while True:
            i = 10
    result = cookies.bake(extra_context={'project_name': 'test_project'})
    assert result.exit_code == 0
    assert result.exception is None
    assert result.project.basename == 'test_project'
    assert result.project.isdir()
    assert result.project.join('README.md').isfile()
    assert result.project.join('template.yaml').isfile()
    assert result.project.join('hello-world').isdir()
    assert result.project.join('hello-world', 'main.go').isfile()
    assert result.project.join('hello-world', 'main_test.go').isfile()

def test_app_content(cookies):
    if False:
        for i in range(10):
            print('nop')
    result = cookies.bake(extra_context={'project_name': 'test_project'})
    app_file = result.project.join('hello-world', 'main.go')
    app_content = app_file.readlines()
    app_content = ''.join(app_content)
    contents = ('github.com/aws/aws-lambda-go/events', 'resp, err := http.Get(DefaultHTTPGetAddress)', 'lambda.Start(handler)')
    for content in contents:
        assert content in app_content

def test_app_test_content(cookies):
    if False:
        for i in range(10):
            print('nop')
    result = cookies.bake(extra_context={'project_name': 'test_project'})
    app_file = result.project.join('hello-world', 'main_test.go')
    app_content = app_file.readlines()
    app_content = ''.join(app_content)
    contents = ('DefaultHTTPGetAddress = "http://127.0.0.1:12345"', 'DefaultHTTPGetAddress = ts.URL', 'Successful Request')
    for content in contents:
        assert content in app_content

def test_app_template_content(cookies):
    if False:
        print('Hello World!')
    result = cookies.bake(extra_context={'project_name': 'test_project'})
    app_file = result.project.join('template.yaml')
    app_content = app_file.readlines()
    app_content = ''.join(app_content)
    contents = ('Runtime: go1.x', 'HelloWorldFunction')
    for content in contents:
        assert content in app_content