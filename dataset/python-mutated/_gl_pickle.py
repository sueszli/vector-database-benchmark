from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from . import util as _util, toolkits as _toolkits, SFrame as _SFrame, SArray as _SArray, SGraph as _SGraph, load_sgraph as _load_graph
from .util import _get_aws_credentials as _util_get_aws_credentials, _cloudpickle
import pickle as _pickle
import uuid as _uuid
import os as _os
import zipfile as _zipfile
import shutil as _shutil
import atexit as _atexit
import glob as _glob

def _get_aws_credentials():
    if False:
        return 10
    (key, secret) = _util_get_aws_credentials()
    return {'aws_access_key_id': key, 'aws_secret_access_key': secret}

def _get_temp_filename():
    if False:
        while True:
            i = 10
    return _util._make_temp_filename(prefix='gl_pickle_')

def _get_tmp_file_location():
    if False:
        print('Hello World!')
    return _util._make_temp_directory(prefix='gl_pickle_')

def _is_not_pickle_safe_gl_model_class(obj_class):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if a Turi create model is pickle safe.\n\n    The function does it by checking that _CustomModel is the base class.\n\n    Parameters\n    ----------\n    obj_class    : Class to be checked.\n\n    Returns\n    ----------\n    True if the GLC class is a model and is pickle safe.\n\n    '
    if issubclass(obj_class, _toolkits._model.CustomModel):
        return not obj_class._is_gl_pickle_safe()
    return False

def _is_not_pickle_safe_gl_class(obj_class):
    if False:
        return 10
    '\n    Check if class is a Turi create model.\n\n    The function does it by checking the method resolution order (MRO) of the\n    class and verifies that _Model is the base class.\n\n    Parameters\n    ----------\n    obj_class    : Class to be checked.\n\n    Returns\n    ----------\n    True if the class is a GLC Model.\n\n    '
    gl_ds = [_SFrame, _SArray, _SGraph]
    return obj_class in gl_ds or _is_not_pickle_safe_gl_model_class(obj_class)

def _get_gl_class_type(obj_class):
    if False:
        return 10
    '\n    Internal util to get the type of the GLC class. The pickle file stores\n    this name so that it knows how to construct the object on unpickling.\n\n    Parameters\n    ----------\n    obj_class    : Class which has to be categorized.\n\n    Returns\n    ----------\n    A class type for the pickle file to save.\n\n    '
    if obj_class == _SFrame:
        return 'SFrame'
    elif obj_class == _SGraph:
        return 'SGraph'
    elif obj_class == _SArray:
        return 'SArray'
    elif _is_not_pickle_safe_gl_model_class(obj_class):
        return 'Model'
    else:
        return None

def _get_gl_object_from_persistent_id(type_tag, gl_archive_abs_path):
    if False:
        while True:
            i = 10
    '\n    Internal util to get a GLC object from a persistent ID in the pickle file.\n\n    Parameters\n    ----------\n    type_tag : The name of the glc class as saved in the GLC pickler.\n\n    gl_archive_abs_path: An absolute path to the GLC archive where the\n                          object was saved.\n\n    Returns\n    ----------\n    The GLC object.\n\n    '
    if type_tag == 'SFrame':
        obj = _SFrame(gl_archive_abs_path)
    elif type_tag == 'SGraph':
        obj = _load_graph(gl_archive_abs_path)
    elif type_tag == 'SArray':
        obj = _SArray(gl_archive_abs_path)
    elif type_tag == 'Model':
        from . import load_model as _load_model
        obj = _load_model(gl_archive_abs_path)
    else:
        raise _pickle.UnpicklingError('Turi pickling Error: Unsupported object. Only SFrames, SGraphs, SArrays, and Models are supported.')
    return obj

class GLPickler(_cloudpickle.CloudPickler):

    def _to_abs_path_set(self, l):
        if False:
            i = 10
            return i + 15
        return set([_os.path.abspath(x) for x in l])
    '\n\n    # GLC pickle works with:\n    #\n    # (1) Regular python objects\n    # (2) SArray\n    # (3) SFrame\n    # (4) SGraph\n    # (5) Models\n    # (6) Any combination of (1) - (5)\n\n    Examples\n    --------\n\n    To pickle a collection of objects into a single file:\n\n    .. sourcecode:: python\n\n        from turicreate.util import gl_pickle\n        import turicreate as tc\n\n        obj = {\'foo\': tc.SFrame([1,2,3]),\n               \'bar\': tc.SArray([1,2,3]),\n               \'foo-bar\': [\'foo-and-bar\', tc.SFrame()]}\n\n        # Setup the GLC pickler\n        pickler = gl_pickle.GLPickler(filename = \'foo-bar\')\n        pickler.dump(obj)\n\n        # The pickler has to be closed to make sure the files get closed.\n        pickler.close()\n\n    To unpickle the collection of objects:\n\n    .. sourcecode:: python\n\n        unpickler = gl_pickle.GLUnpickler(filename = \'foo-bar\')\n        obj = unpickler.load()\n        unpickler.close()\n        print obj\n\n    The GLC pickler needs a temporary working directory to manage GLC objects.\n    This temporary working path must be a local path to the file system. It\n    can also be a relative path in the FS.\n\n    .. sourcecode:: python\n\n        unpickler = gl_pickle.GLUnpickler(\'foo-bar\')\n        obj = unpickler.load()\n        unpickler.close()\n        print obj\n\n\n    Notes\n    --------\n\n    The GLC pickler saves the files into single zip archive with the following\n    file layout.\n\n    pickle_file_name: Name of the file in the archive that contains\n                      the name of the pickle file.\n                      The comment in the ZipFile contains the version number\n                      of the GLC pickler used.\n\n    "pickle_file": The pickle file that stores all python objects. For GLC objects\n                   the pickle file contains a tuple with (ClassName, relative_path)\n                   which stores the name of the GLC object type and a relative\n                   path (in the zip archive) which points to the GLC archive\n                   root directory.\n\n    "gl_archive_dir_1" : A directory which is the GLC archive for a single\n                          object.\n\n     ....\n\n    "gl_archive_dir_N"\n\n\n\n    '

    def __init__(self, filename, protocol=-1, min_bytes_to_save=0):
        if False:
            i = 10
            return i + 15
        '\n\n        Construct a  GLC pickler.\n\n        Parameters\n        ----------\n        filename  : Name of the file to write to. This file is all you need to pickle\n                    all objects (including GLC objects).\n\n        protocol  : Pickle protocol (see pickle docs). Note that all pickle protocols\n                    may not be compatible with GLC objects.\n\n        min_bytes_to_save : Cloud pickle option (see cloud pickle docs).\n\n        Returns\n        ----------\n        GLC pickler.\n\n        '
        self.archive_filename = None
        self.gl_temp_storage_path = _get_tmp_file_location()
        self.gl_object_memo = set()
        self.mark_for_delete = set()
        filename = _os.path.abspath(_os.path.expanduser(_os.path.expandvars(filename)))
        if not _os.path.exists(filename):
            _os.makedirs(filename)
        elif _os.path.isdir(filename):
            self.mark_for_delete = self._to_abs_path_set(_glob.glob(_os.path.join(filename, '*')))
            self.mark_for_delete -= self._to_abs_path_set([_os.path.join(filename, 'pickle_archive'), _os.path.join(filename, 'version')])
        elif _os.path.isfile(filename):
            _os.remove(filename)
            _os.makedirs(filename)
        self.gl_temp_storage_path = filename
        relative_pickle_filename = 'pickle_archive'
        pickle_filename = _os.path.join(self.gl_temp_storage_path, relative_pickle_filename)
        try:
            self.file = open(pickle_filename, 'wb')
            _cloudpickle.CloudPickler.__init__(self, self.file, protocol)
        except IOError as err:
            print('Turi create pickling error: %s' % err)
        with open(_os.path.join(self.gl_temp_storage_path, 'version'), 'w') as f:
            f.write('1.0')

    def dump(self, obj):
        if False:
            while True:
                i = 10
        _cloudpickle.CloudPickler.dump(self, obj)

    def persistent_id(self, obj):
        if False:
            return 10
        '\n        Provide a persistent ID for "saving" GLC objects by reference. Return\n        None for all non GLC objects.\n\n        Parameters\n        ----------\n\n        obj: Name of the object whose persistent ID is extracted.\n\n        Returns\n        --------\n        None if the object is not a GLC object. (ClassName, relative path)\n        if the object is a GLC object.\n\n        Notes\n        -----\n\n        Borrowed from pickle docs (https://docs.python.org/2/library/_pickle.html)\n\n        For the benefit of object persistence, the pickle module supports the\n        notion of a reference to an object outside the pickled data stream.\n\n        To pickle objects that have an external persistent id, the pickler must\n        have a custom persistent_id() method that takes an object as an argument and\n        returns either None or the persistent id for that object.\n\n        For GLC objects, the persistent_id is merely a relative file path (within\n        the ZIP archive) to the GLC archive where the GLC object is saved. For\n        example:\n\n            (SFrame, \'sframe-save-path\')\n            (SGraph, \'sgraph-save-path\')\n            (Model, \'model-save-path\')\n\n        '
        obj_class = None if not hasattr(obj, '__class__') else obj.__class__
        if obj_class is None:
            return None
        if _is_not_pickle_safe_gl_class(obj_class):
            if id(obj) in self.gl_object_memo:
                return (None, None, id(obj))
            else:
                relative_filename = str(_uuid.uuid4())
                filename = _os.path.join(self.gl_temp_storage_path, relative_filename)
                self.mark_for_delete -= set([filename])
                obj.save(filename)
                self.gl_object_memo.add(id(obj))
                return (_get_gl_class_type(obj.__class__), relative_filename, id(obj))
        else:
            return None

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close the pickle file, and the zip archive file. The single zip archive\n        file can now be shipped around to be loaded by the unpickler.\n        '
        if self.file is None:
            return
        self.file.close()
        self.file = None
        for f in self.mark_for_delete:
            error = [False]

            def register_error(*args):
                if False:
                    for i in range(10):
                        print('nop')
                error[0] = True
            _shutil.rmtree(f, onerror=register_error)
            if error[0]:
                _atexit.register(_shutil.rmtree, f, ignore_errors=True)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self.close()

class GLUnpickler(_pickle.Unpickler):
    """
    # GLC unpickler works with a GLC pickler archive or a regular pickle
    # archive.
    #
    # Works with
    # (1) GLPickler archive
    # (2) Cloudpickle archive
    # (3) Python pickle archive

    Examples
    --------
    To unpickle the collection of objects:

    .. sourcecode:: python

        unpickler = gl_pickle.GLUnpickler('foo-bar')
        obj = unpickler.load()
        print obj

    """

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        '\n        Construct a GLC unpickler.\n\n        Parameters\n        ----------\n        filename  : Name of the file to read from. The file can be a GLC pickle\n                    file, a cloud pickle file, or a python pickle file.\n        Returns\n        ----------\n        GLC unpickler.\n        '
        self.gl_object_memo = {}
        self.pickle_filename = None
        self.tmp_file = None
        self.file = None
        self.gl_temp_storage_path = _get_tmp_file_location()
        self.directory_mode = True
        filename = _os.path.abspath(_os.path.expanduser(_os.path.expandvars(filename)))
        if not _os.path.exists(filename):
            raise IOError('%s is not a valid file name.' % filename)
        if _zipfile.is_zipfile(filename):
            self.directory_mode = False
            pickle_filename = None
            zf = _zipfile.ZipFile(filename, allowZip64=True)
            for info in zf.infolist():
                if info.filename == 'pickle_file':
                    pickle_filename = zf.read(info.filename).decode()
            if pickle_filename is None:
                raise IOError('Cannot pickle file of the given format. File must be one of (a) GLPickler archive, (b) Cloudpickle archive, or (c) python pickle archive.')
            try:
                outpath = self.gl_temp_storage_path
                zf.extractall(outpath)
            except IOError as err:
                print('Turi pickle extraction error: %s ' % err)
            self.pickle_filename = _os.path.join(self.gl_temp_storage_path, pickle_filename)
        elif _os.path.isdir(filename):
            self.directory_mode = True
            pickle_filename = _os.path.join(filename, 'pickle_archive')
            if not _os.path.exists(pickle_filename):
                raise IOError('Corrupted archive: Missing pickle file %s.' % pickle_filename)
            if not _os.path.exists(_os.path.join(filename, 'version')):
                raise IOError('Corrupted archive: Missing version file.')
            self.pickle_filename = pickle_filename
            self.gl_temp_storage_path = _os.path.abspath(filename)
        else:
            self.directory_mode = False
            self.pickle_filename = filename
        self.file = open(self.pickle_filename, 'rb')
        _pickle.Unpickler.__init__(self, self.file)

    def persistent_load(self, pid):
        if False:
            return 10
        '\n        Reconstruct a GLC object using the persistent ID.\n\n        This method should not be used externally. It is required by the unpickler super class.\n\n        Parameters\n        ----------\n        pid      : The persistent ID used in pickle file to save the GLC object.\n\n        Returns\n        ----------\n        The GLC object.\n        '
        if len(pid) == 2:
            (type_tag, filename) = pid
            abs_path = _os.path.join(self.gl_temp_storage_path, filename)
            return _get_gl_object_from_persistent_id(type_tag, abs_path)
        else:
            (type_tag, filename, object_id) = pid
            if object_id in self.gl_object_memo:
                return self.gl_object_memo[object_id]
            else:
                abs_path = _os.path.join(self.gl_temp_storage_path, filename)
                obj = _get_gl_object_from_persistent_id(type_tag, abs_path)
                self.gl_object_memo[object_id] = obj
                return obj

    def close(self):
        if False:
            return 10
        '\n        Clean up files that were created.\n        '
        if self.file:
            self.file.close()
            self.file = None
        if self.tmp_file and _os.path.isfile(self.tmp_file):
            _os.remove(self.tmp_file)
            self.tmp_file = None

    def __del__(self):
        if False:
            while True:
                i = 10
        '\n        Clean up files that were created.\n        '
        self.close()