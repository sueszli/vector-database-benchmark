from dagster_pandas.constraints import all_unique_validator, categorical_column_validator_factory, column_range_validation_factory, dtype_in_set_validation_factory, non_null_validation, nonnull
from numpy import nan as NaN

def test_unique():
    if False:
        while True:
            i = 10
    testlst = [0, 1, 2]
    assert all_unique_validator(testlst)[0]
    faillst = [0, 0, 1]
    assert not all_unique_validator(faillst)[0]

def test_ignore_vals():
    if False:
        i = 10
        return i + 15
    faillst = [0, NaN, NaN]
    assert all_unique_validator(faillst, ignore_missing_vals=True)[0]

def test_null():
    if False:
        for i in range(10):
            print('nop')
    testval = NaN
    assert not non_null_validation(testval)[0]

def test_range():
    if False:
        return 10
    testfunc = column_range_validation_factory(minim=0, maxim=10)
    assert testfunc(1)[0]
    assert not testfunc(20)[0]

def test_dtypes():
    if False:
        i = 10
        return i + 15
    testfunc = dtype_in_set_validation_factory((int, float))
    assert testfunc(1)[0]
    assert testfunc(1.5)[0]
    assert not testfunc('a')[0]

def test_nonnull():
    if False:
        i = 10
        return i + 15
    testfunc = dtype_in_set_validation_factory((int, float))
    assert testfunc(NaN)[0]
    ntestfunc = nonnull(testfunc)
    assert not ntestfunc(NaN)[0]

def test_categorical():
    if False:
        i = 10
        return i + 15
    testfunc = categorical_column_validator_factory(['a', 'b'], ignore_missing_vals=True)
    assert testfunc('a')[0]
    assert testfunc('b')[0]
    assert testfunc(NaN)[0]
    assert not testfunc('c')[0]