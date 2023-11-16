import package.Something

def test_func():
    if False:
        return 10
    assert package.Something.calledByTest() == 42