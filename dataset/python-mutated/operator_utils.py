"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect

class Operator(object):
    """Base class for operators in TPOT."""
    root = False
    import_hash = None
    sklearn_class = None
    arg_types = None

class ARGType(object):
    """Base class for parameter specifications."""
    pass

def source_decode(sourcecode, verbose=0):
    if False:
        i = 10
        return i + 15
    "Decode operator source and import operator class.\n\n    Parameters\n    ----------\n    sourcecode: string\n        a string of operator source (e.g 'sklearn.feature_selection.RFE')\n    verbose: int, optional (default: 0)\n        How much information TPOT communicates while it's running.\n        0 = none, 1 = minimal, 2 = high, 3 = all.\n        if verbose > 2 then ImportError will rasie during initialization\n\n\n    Returns\n    -------\n    import_str: string\n        a string of operator class source (e.g. 'sklearn.feature_selection')\n    op_str: string\n        a string of operator class (e.g. 'RFE')\n    op_obj: object\n        operator class (e.g. RFE)\n\n    "
    tmp_path = sourcecode.split('.')
    op_str = tmp_path.pop()
    import_str = '.'.join(tmp_path)
    try:
        if sourcecode.startswith('tpot.'):
            exec('from {} import {}'.format(import_str[4:], op_str))
        else:
            exec('from {} import {}'.format(import_str, op_str))
        op_obj = eval(op_str)
    except Exception as e:
        if verbose > 2:
            raise ImportError('Error: could not import {}.\n{}'.format(sourcecode, e))
        else:
            print('Warning: {} is not available and will not be used by TPOT.'.format(sourcecode))
        op_obj = None
    return (import_str, op_str, op_obj)

def set_sample_weight(pipeline_steps, sample_weight=None):
    if False:
        print('Hello World!')
    'Recursively iterates through all objects in the pipeline and sets sample weight.\n\n    Parameters\n    ----------\n    pipeline_steps: array-like\n        List of (str, obj) tuples from a scikit-learn pipeline or related object\n    sample_weight: array-like\n        List of sample weight\n    Returns\n    -------\n    sample_weight_dict:\n        A dictionary of sample_weight\n\n    '
    sample_weight_dict = {}
    if not isinstance(sample_weight, type(None)):
        for (pname, obj) in pipeline_steps:
            if inspect.getargspec(obj.fit).args.count('sample_weight'):
                step_sw = pname + '__sample_weight'
                sample_weight_dict[step_sw] = sample_weight
    if sample_weight_dict:
        return sample_weight_dict
    else:
        return None

def _is_selector(estimator):
    if False:
        print('Hello World!')
    selector_attributes = ['get_support', 'transform', 'inverse_transform', 'fit_transform']
    return all((hasattr(estimator, attr) for attr in selector_attributes))

def _is_transformer(estimator):
    if False:
        print('Hello World!')
    return hasattr(estimator, 'fit_transform')

def _is_resampler(estimator):
    if False:
        for i in range(10):
            print('nop')
    return hasattr(estimator, 'fit_resample')

def ARGTypeClassFactory(classname, prange, BaseClass=ARGType):
    if False:
        while True:
            i = 10
    'Dynamically create parameter type class.\n\n    Parameters\n    ----------\n    classname: string\n        parameter name in a operator\n    prange: list\n        list of values for the parameter in a operator\n    BaseClass: Class\n        inherited BaseClass for parameter\n\n    Returns\n    -------\n    Class\n        parameter class\n\n    '
    return type(classname, (BaseClass,), {'values': prange})

def TPOTOperatorClassFactory(opsourse, opdict, BaseClass=Operator, ArgBaseClass=ARGType, verbose=0):
    if False:
        i = 10
        return i + 15
    "Dynamically create operator class.\n\n    Parameters\n    ----------\n    opsourse: string\n        operator source in config dictionary (key)\n    opdict: dictionary\n        operator params in config dictionary (value)\n    regression: bool\n        True if it can be used in TPOTRegressor\n    classification: bool\n        True if it can be used in TPOTClassifier\n    BaseClass: Class\n        inherited BaseClass for operator\n    ArgBaseClass: Class\n        inherited BaseClass for parameter\n    verbose: int, optional (default: 0)\n        How much information TPOT communicates while it's running.\n        0 = none, 1 = minimal, 2 = high, 3 = all.\n        if verbose > 2 then ImportError will rasie during initialization\n\n    Returns\n    -------\n    op_class: Class\n        a new class for a operator\n    arg_types: list\n        a list of parameter class\n\n    "
    class_profile = {}
    dep_op_list = {}
    dep_op_type = {}
    (import_str, op_str, op_obj) = source_decode(opsourse, verbose=verbose)
    if not op_obj:
        return (None, None)
    else:
        if is_classifier(op_obj):
            class_profile['root'] = True
            optype = 'Classifier'
        elif is_regressor(op_obj):
            class_profile['root'] = True
            optype = 'Regressor'
        elif _is_selector(op_obj):
            optype = 'Selector'
        elif _is_transformer(op_obj):
            optype = 'Transformer'
        elif _is_resampler(op_obj):
            optype = 'Resampler'
        else:
            raise ValueError('optype must be one of: Classifier, Regressor, Selector, Transformer, or Resampler')

        @classmethod
        def op_type(cls):
            if False:
                return 10
            'Return the operator type.\n\n            Possible values:\n                "Classifier", "Regressor", "Selector", "Transformer"\n            '
            return optype
        class_profile['type'] = op_type
        class_profile['sklearn_class'] = op_obj
        import_hash = {}
        import_hash[import_str] = [op_str]
        arg_types = []
        for pname in sorted(opdict.keys()):
            prange = opdict[pname]
            if not isinstance(prange, dict):
                classname = '{}__{}'.format(op_str, pname)
                arg_types.append(ARGTypeClassFactory(classname, prange, ArgBaseClass))
            else:
                for (dkey, dval) in prange.items():
                    (dep_import_str, dep_op_str, dep_op_obj) = source_decode(dkey, verbose=verbose)
                    if dep_import_str in import_hash:
                        import_hash[dep_import_str].append(dep_op_str)
                    else:
                        import_hash[dep_import_str] = [dep_op_str]
                    dep_op_list[pname] = dep_op_str
                    dep_op_type[pname] = dep_op_obj
                    if dval:
                        for dpname in sorted(dval.keys()):
                            dprange = dval[dpname]
                            classname = '{}__{}__{}'.format(op_str, dep_op_str, dpname)
                            arg_types.append(ARGTypeClassFactory(classname, dprange, ArgBaseClass))
        class_profile['arg_types'] = tuple(arg_types)
        class_profile['import_hash'] = import_hash
        class_profile['dep_op_list'] = dep_op_list
        class_profile['dep_op_type'] = dep_op_type

        @classmethod
        def parameter_types(cls):
            if False:
                i = 10
                return i + 15
            'Return the argument and return types of an operator.\n\n            Parameters\n            ----------\n            None\n\n            Returns\n            -------\n            parameter_types: tuple\n                Tuple of the DEAP parameter types and the DEAP return type for the\n                operator\n\n            '
            return ([np.ndarray] + arg_types, np.ndarray)
        class_profile['parameter_types'] = parameter_types

        @classmethod
        def export(cls, *args):
            if False:
                print('Hello World!')
            'Represent the operator as a string so that it can be exported to a file.\n\n            Parameters\n            ----------\n            args\n                Arbitrary arguments to be passed to the operator\n\n            Returns\n            -------\n            export_string: str\n                String representation of the sklearn class with its parameters in\n                the format:\n                SklearnClassName(param1="val1", param2=val2)\n\n            '
            op_arguments = []
            if dep_op_list:
                dep_op_arguments = {}
                for dep_op_str in dep_op_list.values():
                    dep_op_arguments[dep_op_str] = []
            for (arg_class, arg_value) in zip(arg_types, args):
                aname_split = arg_class.__name__.split('__')
                if isinstance(arg_value, str):
                    arg_value = '"{}"'.format(arg_value)
                if len(aname_split) == 2:
                    op_arguments.append('{}={}'.format(aname_split[-1], arg_value))
                else:
                    dep_op_arguments[aname_split[1]].append('{}={}'.format(aname_split[-1], arg_value))
            tmp_op_args = []
            if dep_op_list:
                for (dep_op_pname, dep_op_str) in dep_op_list.items():
                    arg_value = dep_op_str
                    doptype = dep_op_type[dep_op_pname]
                    if inspect.isclass(doptype):
                        if issubclass(doptype, BaseEstimator) or is_classifier(doptype) or is_regressor(doptype) or _is_transformer(doptype) or _is_resampler(doptype) or issubclass(doptype, Kernel):
                            arg_value = '{}({})'.format(dep_op_str, ', '.join(dep_op_arguments[dep_op_str]))
                    tmp_op_args.append('{}={}'.format(dep_op_pname, arg_value))
            op_arguments = tmp_op_args + op_arguments
            return '{}({})'.format(op_obj.__name__, ', '.join(op_arguments))
        class_profile['export'] = export
        op_classname = 'TPOT_{}'.format(op_str)
        op_class = type(op_classname, (BaseClass,), class_profile)
        op_class.__name__ = op_str
        return (op_class, arg_types)