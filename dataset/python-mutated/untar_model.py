import sys
import tarfile

def untar(fname, dirs):
    if False:
        for i in range(10):
            print('nop')
    '\n    extract the tar.gz file\n    :param fname: the name of tar.gz file\n    :param dirs: the path of decompressed file\n    :return: bool\n    '
    try:
        t = tarfile.open(name=fname, mode='r:gz')
        t.extractall(path=dirs)
        return True
    except Exception as e:
        print(e)
        return False
untar(sys.argv[1], sys.argv[2])