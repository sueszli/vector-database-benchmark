from joblib.compressor import BinaryZlibFile
from joblib.testing import parametrize

@parametrize('filename', ['test', u'test'])
def test_binary_zlib_file(tmpdir, filename):
    if False:
        i = 10
        return i + 15
    'Testing creation of files depending on the type of the filenames.'
    binary_file = BinaryZlibFile(tmpdir.join(filename).strpath, mode='wb')
    binary_file.close()