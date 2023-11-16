"""Test the old numpy pickler, compatibility version."""
from joblib import numpy_pickle_compat

def test_z_file(tmpdir):
    if False:
        while True:
            i = 10
    filename = tmpdir.join('test.pkl').strpath
    data = numpy_pickle_compat.asbytes('Foo, \n Bar, baz, \n\nfoobar')
    with open(filename, 'wb') as f:
        numpy_pickle_compat.write_zfile(f, data)
    with open(filename, 'rb') as f:
        data_read = numpy_pickle_compat.read_zfile(f)
    assert data == data_read