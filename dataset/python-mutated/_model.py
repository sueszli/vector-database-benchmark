"""
@package turicreate.toolkits

Defines a basic interface for a model object.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
import turicreate._connect.main as glconnect
from turicreate.data_structures.sframe import SFrame as _SFrame
import turicreate.extensions as _extensions
from turicreate.extensions import _wrap_function_return
from turicreate.toolkits._internal_utils import _toolkit_serialize_summary_struct
from turicreate.util import _make_internal_url
from turicreate.toolkits._main import ToolkitError
from turicreate._deps.minimal_package import _minimal_package_import_check
import turicreate.util._file_util as file_util
import os
from copy import copy as _copy
import six as _six
import warnings
MODEL_NAME_MAP = {}

def load_model(location):
    if False:
        i = 10
        return i + 15
    "\n    Load any Turi Create model that was previously saved.\n\n    This function assumes the model (can be any model) was previously saved in\n    Turi Create model format with model.save(filename).\n\n    Parameters\n    ----------\n    location : string\n        Location of the model to load. Can be a local path or a remote URL.\n        Because models are saved as directories, there is no file extension.\n\n    Examples\n    ----------\n    >>> model.save('my_model_file')\n    >>> loaded_model = tc.load_model('my_model_file')\n    "
    protocol = file_util.get_protocol(location)
    dir_archive_exists = False
    if protocol == '':
        model_path = file_util.expand_full_path(location)
        dir_archive_exists = file_util.exists(os.path.join(model_path, 'dir_archive.ini'))
    else:
        model_path = location
        if protocol in ['http', 'https', 's3']:
            dir_archive_exists = True
        else:
            import posixpath
            dir_archive_exists = file_util.exists(posixpath.join(model_path, 'dir_archive.ini'))
    if not dir_archive_exists:
        raise IOError('Directory %s does not exist' % location)
    _internal_url = _make_internal_url(location)
    saved_state = glconnect.get_unity().load_model(_internal_url)
    saved_state = _wrap_function_return(saved_state)
    key = u'archive_version'
    archive_version = saved_state[key] if key in saved_state else saved_state[key.encode()]
    if archive_version < 0:
        raise ToolkitError('File does not appear to be a Turi Create model.')
    elif archive_version > 1:
        raise ToolkitError('Unable to load model.\n\nThis model looks to have been saved with a future version of Turi Create.\nPlease upgrade Turi Create before attempting to load this model file.')
    elif archive_version == 1:
        name = saved_state['model_name']
        if name in MODEL_NAME_MAP:
            cls = MODEL_NAME_MAP[name]
            if 'model' in saved_state:
                if name in ['activity_classifier', 'object_detector', 'style_transfer', 'drawing_classifier']:
                    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
                return cls(saved_state['model'])
            else:
                model_data = saved_state['side_data']
                model_version = model_data['model_version']
                del model_data['model_version']
                if name == 'activity_classifier':
                    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
                    model = _extensions.activity_classifier()
                    model.import_from_custom_model(model_data, model_version)
                    return cls(model)
                if name == 'object_detector':
                    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
                    model = _extensions.object_detector()
                    model.import_from_custom_model(model_data, model_version)
                    return cls(model)
                if name == 'style_transfer':
                    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
                    model = _extensions.style_transfer()
                    model.import_from_custom_model(model_data, model_version)
                    return cls(model)
                if name == 'drawing_classifier':
                    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
                    model = _extensions.drawing_classifier()
                    model.import_from_custom_model(model_data, model_version)
                    return cls(model)
                if name == 'one_shot_object_detector':
                    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
                    od_cls = MODEL_NAME_MAP['object_detector']
                    if 'detector_model' in model_data['detector']:
                        model_data['detector'] = od_cls(model_data['detector']['detector_model'])
                    else:
                        model = _extensions.object_detector()
                        model.import_from_custom_model(model_data['detector'], model_data['_detector_version'])
                        model_data['detector'] = od_cls(model)
                    return cls(model_data)
                return cls._load_version(model_data, model_version)
        elif hasattr(_extensions, name):
            return saved_state['model']
        else:
            raise ToolkitError("Unable to load model of name '%s'; model name not registered." % name)
    else:
        import sys
        sys.stderr.write('This model was saved in a legacy model format. Compatibility cannot be guaranteed in future versions.\n')
        if _six.PY3:
            raise ToolkitError('Unable to load legacy model in Python 3.\n\nTo migrate a model, try loading it using Turi Create 4.0 or\nlater in Python 2 and then re-save it. The re-saved model should\nwork in Python 3.')
        if 'graphlab' not in sys.modules:
            sys.modules['graphlab'] = sys.modules['turicreate']
            sys.modules['turicreate_util'] = sys.modules['turicreate.util']
            sys.modules['graphlab_util'] = sys.modules['turicreate.util']
            for (k, v) in list(sys.modules.items()):
                if 'turicreate' in k:
                    sys.modules[k.replace('turicreate', 'graphlab')] = v
        import pickle
        model_wrapper = pickle.loads(saved_state[b'model_wrapper'])
        return model_wrapper(saved_state[b'model_base'])

def _get_default_options_wrapper(unity_server_model_name, module_name='', python_class_name='', sdk_model=False):
    if False:
        print('Hello World!')
    "\n    Internal function to return a get_default_options function.\n\n    Parameters\n    ----------\n    unity_server_model_name: str\n        Name of the class/toolkit as registered with the unity server\n\n    module_name: str, optional\n        Name of the module.\n\n    python_class_name: str, optional\n        Name of the Python class.\n\n    sdk_model : bool, optional (default False)\n        True if the SDK interface was used for the model. False otherwise.\n\n    Examples\n    ----------\n    get_default_options = _get_default_options_wrapper('classifier_svm',\n                                                       'svm', 'SVMClassifier')\n    "

    def get_default_options_for_model(output_type='sframe'):
        if False:
            print('Hello World!')
        "\n        Get the default options for the toolkit\n        :class:`~turicreate.{module_name}.{python_class_name}`.\n\n        Parameters\n        ----------\n        output_type : str, optional\n\n            The output can be of the following types.\n\n            - `sframe`: A table description each option used in the model.\n            - `json`: A list of option dictionaries suitable for JSON serialization.\n\n            | Each dictionary/row in the dictionary/SFrame object describes the\n              following parameters of the given model.\n\n            +------------------+-------------------------------------------------------+\n            |      Name        |                  Description                          |\n            +==================+=======================================================+\n            | name             | Name of the option used in the model.                 |\n            +------------------+---------+---------------------------------------------+\n            | description      | A detailed description of the option used.            |\n            +------------------+-------------------------------------------------------+\n            | type             | Option type (REAL, BOOL, INTEGER or CATEGORICAL)      |\n            +------------------+-------------------------------------------------------+\n            | default_value    | The default value for the option.                     |\n            +------------------+-------------------------------------------------------+\n            | possible_values  | List of acceptable values (CATEGORICAL only)          |\n            +------------------+-------------------------------------------------------+\n            | lower_bound      | Smallest acceptable value for this option (REAL only) |\n            +------------------+-------------------------------------------------------+\n            | upper_bound      | Largest acceptable value for this option (REAL only)  |\n            +------------------+-------------------------------------------------------+\n\n        Returns\n        -------\n        out : dict/SFrame\n\n        See Also\n        --------\n        turicreate.{module_name}.{python_class_name}.get_current_options\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n          >>> import turicreate\n\n          # SFrame formatted output.\n          >>> out_sframe = turicreate.{module_name}.get_default_options()\n\n          # dict formatted output suitable for JSON serialization.\n          >>> out_json = turicreate.{module_name}.get_default_options('json')\n        "
        if sdk_model:
            response = _tc.extensions._toolkits_sdk_get_default_options(unity_server_model_name)
        else:
            response = _tc.extensions._toolkits_get_default_options(unity_server_model_name)
        if output_type == 'json':
            return response
        else:
            json_list = [{'name': k, '': v} for (k, v) in response.items()]
            return _SFrame(json_list).unpack('X1', column_name_prefix='').unpack('X1', column_name_prefix='')
    get_default_options_for_model.__doc__ = get_default_options_for_model.__doc__.format(python_class_name=python_class_name, module_name=module_name)
    return get_default_options_for_model

class RegistrationMetaClass(type):

    def __new__(meta, name, bases, class_dict):
        if False:
            for i in range(10):
                print('nop')
        global MODEL_NAME_MAP
        cls = type.__new__(meta, name, bases, class_dict)
        if name == 'Model' or name == 'CustomModel':
            return cls
        native_name = cls._native_name()
        if isinstance(native_name, (list, tuple)):
            for i in native_name:
                MODEL_NAME_MAP[i] = cls
        elif native_name is not None:
            MODEL_NAME_MAP[native_name] = cls
        return cls

class PythonProxy(object):
    """
    Simple wrapper around a Python dict that exposes get/list_fields to emulate
    a "proxy" object entirely from Python.
    """

    def __init__(self, state={}):
        if False:
            for i in range(10):
                print('nop')
        self.state = _copy(state)

    def get(self, key):
        if False:
            i = 10
            return i + 15
        return self.state[key]

    def keys(self):
        if False:
            i = 10
            return i + 15
        return self.state.keys()

    def list_fields(self):
        if False:
            i = 10
            return i + 15
        return list(self.state.keys())

    def __contains__(self, key):
        if False:
            return 10
        return self.state.__contains__(key)

    def __getitem__(self, field):
        if False:
            print('Hello World!')
        return self.state[field]

    def __setitem__(self, key, value):
        if False:
            return 10
        self.state[key] = value

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        del self.state[key]

    def pop(self, key):
        if False:
            while True:
                i = 10
        return self.state.pop(key)

    def update(self, d):
        if False:
            while True:
                i = 10
        self.state.update(d)

    def get_state(self):
        if False:
            while True:
                i = 10
        return _copy(self.state)

class ExposeAttributesFromProxy(object):
    """Mixin to use when a __proxy__ class attribute should be used for
    additional fields. This allows tab-complete (i.e., calling __dir__ on the
    object) to include class methods as well as the results of
    __proxy__.list_fields().
    """
    'The UnityModel Proxy Object'
    __proxy__ = None

    def __dir__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Combine the results of dir from the current class with the results of\n        list_fields().\n        '
        return dir(self.__class__) + list(self._list_fields()) + ['_list_fields']

    def _get(self, field):
        if False:
            while True:
                i = 10
        "\n        Return the value contained in the model's ``field``.\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out\n            Value of the requested field.\n\n        See Also\n        --------\n        list_fields\n        "
        try:
            return self.__proxy__[field]
        except:
            raise ValueError('There is no model field called {}'.format(field))

    def __getattribute__(self, attr):
        if False:
            while True:
                i = 10
        '\n        Use the internal proxy object for obtaining list_fields.\n        '
        proxy = object.__getattribute__(self, '__proxy__')
        if proxy is None:
            return object.__getattribute__(self, attr)
        if not hasattr(proxy, 'list_fields'):
            fields = []
        else:
            fields = proxy.list_fields()

        def list_fields():
            if False:
                for i in range(10):
                    print('nop')
            return fields
        if attr == '_list_fields':
            return list_fields
        elif attr in fields:
            return self._get(attr)
        else:
            return object.__getattribute__(self, attr)

@_six.add_metaclass(RegistrationMetaClass)
class Model(ExposeAttributesFromProxy):
    """
    This class defines the minimal interface of a model object which is
    backed by a C++ model implementation.

    All state in a Model must be stored in the C++-side __proxy__ object.

    _native_name must be implemented. _native_name can returns a list if there
    are multiple C++ types for the same Python object. The native names *must*
    match the registered name of the model (name() method)

    The constructor must also permit construction from only 1 argument : the proxy object.

    For instance:

    class MyModel:
        @classmethod
        def _native_name(cls):
            return "MyModel"

    Or:

    class NearestNeighborsModel:
        @classmethod
        def _native_name(cls):
            return ["nearest_neighbors_ball_tree", "nearest_neighbors_brute_force", "nearest_neighbors_lsh"]
    """

    def _name(self):
        if False:
            return 10
        '\n        Returns the name of the model class.\n\n        Returns\n        -------\n        out : str\n            The name of the model class.\n\n        Examples\n        --------\n        >>> model_name = m._name()\n        '
        return self.__class__.__name__

    def _get(self, field):
        if False:
            for i in range(10):
                print('nop')
        "Return the value for the queried field.\n\n        Each of these fields can be queried in one of two ways:\n\n        >>> out = m['field']\n        >>> out = m.get('field')  # equivalent to previous line\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out : value\n            The current value of the requested field.\n\n        "
        if field in self._list_fields():
            return self.__proxy__.get_value(field)
        else:
            raise KeyError('Field "%s" not in model. Available fields are %s.' % (field, ', '.join(self._list_fields())))

    @classmethod
    def _native_name(cls):
        if False:
            print('Hello World!')
        raise NotImplementedError('_native_name not implemented')

    def save(self, location):
        if False:
            return 10
        "\n        Save the model. The model is saved as a directory which can then be\n        loaded using the :py:func:`~turicreate.load_model` method.\n\n        Parameters\n        ----------\n        location : string\n            Target destination for the model. Can be a local path or remote URL.\n\n        See Also\n        ----------\n        turicreate.load_model\n\n        Examples\n        ----------\n        >>> model.save('my_model_file')\n        >>> loaded_model = turicreate.load_model('my_model_file')\n        "
        return glconnect.get_unity().save_model(self, _make_internal_url(location))

    def summary(self, output=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Print a summary of the model. The summary includes a description of\n        training data, options, hyper-parameters, and statistics measured\n        during model creation.\n\n        Parameters\n        ----------\n        output : str, None\n            The type of summary to return.\n\n            - None or 'stdout' : print directly to stdout.\n\n            - 'str' : string of summary\n\n            - 'dict' : a dict with 'sections' and 'section_titles' ordered\n              lists. The entries in the 'sections' list are tuples of the form\n              ('label', 'value').\n\n        Examples\n        --------\n        >>> m.summary()\n        "
        if output is None or output == 'stdout':
            try:
                print(self.__repr__())
            except:
                return self.__class__.__name__
        elif output == 'str':
            return self.__repr__()
        elif output == 'dict':
            return _toolkit_serialize_summary_struct(self, *self._get_summary_struct())
        else:
            raise ToolkitError('Unsupported argument ' + str(output) + ' for "summary" parameter.')

    def __repr__(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.__repr__()

@_six.add_metaclass(RegistrationMetaClass)
class CustomModel(ExposeAttributesFromProxy):
    """
    This class is used to implement Python-only models.

    The following must be implemented
    - _get_version
    - _get_native_state
    - _native_name (class method)
    - _load_version (class method)

    On save, get_native_state is called which must return a dictionary
    containing the state of the object. This must contain
    all the relevant information needed to reconstruct the model.

    On load _load_version is used to reconstruct the object.

    _native_name must return a globally unique name. This is the name used to
    identify the model.

    Example
    -------
    class MyModelMinimal(CustomModel):
        def __init__(self, prediction):
            # We use PythonProxy here so that we get tab completion
            self.__proxy__ = PythonProxy(state)

        @classmethod
        def create(cls, prediction):
            return MyModelMinimal({'prediction':prediction})

        def predict(self):
            return self.__proxy__['prediction']

        def _get_version(self):
            return 0

        @classmethod
        def _native_name(cls):
            return "MyModelMinimal"

        def _get_native_state(self):
            # return a dictionary completely describing the object
            return self.__proxy__.get_state()

        @classmethod
        def _load_version(cls, state, version):
            # loads back from a dictionary
            return MyModelMinimal(state)


    # a wrapper around logistic classifier
    class MyModelComplicated(CustomModel):
        def __init__(self, state):
            # We use PythonProxy here so that we get tab completion
            self.__proxy__ = PythonProxy(state)

        @classmethod
        def create(cls, sf, target):
            classifier = tc.logistic_classifier.create(sf, target=target)
            state = {'classifier':classifier, 'target':target}
            return MyModelComplicated(state)

        def predict(self, sf):
            return self.__proxy__['classifier'].predict(sf)

        def _get_version(self):
            return 0

        @classmethod
        def _native_name(cls):
            return "MyModelComplicated"

        def _get_native_state(self):
            # make sure to not accidentally modify the proxy object.
            # take a copy of it.
            state = self.__proxy__.get_state()

            # We don't know how to serialize a Python class, hence we need to
            # reduce the classifier to the proxy object before saving it.
            state['classifier'] = state['classifier'].__proxy__
            return state

        @classmethod
        def _load_version(cls, state, version):
            assert(version == 0)
            # we need to undo what we did at save and turn the proxy object
            # back into a Python class
            state['classifier'] = LogisticClassifier(state['classifier'])
            return MyModelComplicated(state)


    # Construct the model
    >>> custom_model = MyModel(sf, glc_model)

    ## The model can be saved and loaded like any Turi Create model.
    >>> model.save('my_model_file')
    >>> loaded_model = tc.load_model('my_model_file')
    """

    def __init__(self):
        if False:
            return 10
        pass

    def name(self):
        if False:
            print('Hello World!')
        '\n        Returns the name of the model.\n\n        ..WARNING:: This function is deprecated, It will be removed in the next\n        release. Use type(CustomModel) instead.\n\n        Returns\n        -------\n        out : str\n            The name of the model object.\n\n        Examples\n        --------\n        >>> model_name = m.name()\n        '
        warnings.warn("This function is deprecated. It will be removed in the next release. Please use python's builtin type function instead.")
        return self.__class__.__name__

    def summary(self, output=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Print a summary of the model. The summary includes a description of\n        training data, options, hyper-parameters, and statistics measured\n        during model creation.\n\n        Parameters\n        ----------\n        output : str, None\n            The type of summary to return.\n\n            - None or 'stdout' : print directly to stdout.\n\n            - 'str' : string of summary\n\n            - 'dict' : a dict with 'sections' and 'section_titles' ordered\n              lists. The entries in the 'sections' list are tuples of the form\n              ('label', 'value').\n\n        Examples\n        --------\n        >>> m.summary()\n        "
        if output is None or output == 'stdout':
            try:
                print(self.__repr__())
            except:
                return self.__class__.__name__
        elif output == 'str':
            return self.__repr__()
        elif output == 'dict':
            return _toolkit_serialize_summary_struct(self, *self._get_summary_struct())
        else:
            raise ToolkitError('Unsupported argument ' + str(output) + ' for "summary" parameter.')

    def _get_version(self):
        if False:
            return 10
        raise NotImplementedError('_get_version not implemented')

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self._get(key)

    def _get_native_state(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('_get_native_state not implemented')

    def save(self, location):
        if False:
            for i in range(10):
                print('nop')
        "\n        Save the model. The model is saved as a directory which can then be\n        loaded using the :py:func:`~turicreate.load_model` method.\n\n        Parameters\n        ----------\n        location : string\n            Target destination for the model. Can be a local path or remote URL.\n\n        See Also\n        ----------\n        turicreate.load_model\n\n        Examples\n        ----------\n        >>> model.save('my_model_file')\n        >>> loaded_model = tc.load_model('my_model_file')\n\n        "
        import copy
        state = copy.copy(self._get_native_state())
        state['model_version'] = self._get_version()
        return glconnect.get_unity().save_model2(self.__class__._native_name(), _make_internal_url(location), state)

    @classmethod
    def _native_name(cls):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('native_name')

    @classmethod
    def _load_version(cls, state, version):
        if False:
            return 10
        '\n        An function to load an object with a specific version of the class.\n\n        WARNING: This implementation is very simple.\n                 Overwrite for smarter implementations.\n\n        Parameters\n        ----------\n        state : dict\n            The saved state object\n\n        version : int\n            A version number as obtained from _get_version()\n        '
        raise NotImplementedError('load')