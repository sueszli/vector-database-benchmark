from dagster._utils import camelcase

def test_camelcase():
    if False:
        for i in range(10):
            print('nop')
    assert camelcase('foo') == 'Foo'
    assert camelcase('foo_bar') == 'FooBar'
    assert camelcase('foo.bar') == 'FooBar'
    assert camelcase('foo-bar') == 'FooBar'
    assert camelcase('') == ''