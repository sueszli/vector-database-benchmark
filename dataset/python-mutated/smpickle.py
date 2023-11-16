"""Helper files for pickling"""
from statsmodels.iolib.openfile import get_file_obj

def save_pickle(obj, fname):
    if False:
        print('Hello World!')
    '\n    Save the object to file via pickling.\n\n    Parameters\n    ----------\n    fname : {str, pathlib.Path}\n        Filename to pickle to\n    '
    import pickle
    with get_file_obj(fname, 'wb') as fout:
        pickle.dump(obj, fout, protocol=-1)

def load_pickle(fname):
    if False:
        return 10
    '\n    Load a previously saved object\n\n    .. warning::\n\n       Loading pickled models is not secure against erroneous or maliciously\n       constructed data. Never unpickle data received from an untrusted or\n       unauthenticated source.\n\n    Parameters\n    ----------\n    fname : {str, pathlib.Path}\n        Filename to unpickle\n\n    Notes\n    -----\n    This method can be used to load *both* models and results.\n    '
    import pickle
    with get_file_obj(fname, 'rb') as fin:
        return pickle.load(fin)