""" This module maintains the parameter specification classes.

These are used for function, lambdas, generators. They are also a factory
for the respective variable objects. One of the difficulty of Python and
its parameter parsing is that they are allowed to be nested like this:

(a,b), c

Much like in assignments, which are very similar to parameters, except
that parameters may also be assigned from a dictionary, they are no less
flexible.

"""
from nuitka import Variables
from nuitka.PythonVersions import python_version
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances

class TooManyArguments(Exception):

    def __init__(self, real_exception):
        if False:
            for i in range(10):
                print('nop')
        Exception.__init__(self)
        self.real_exception = real_exception

    def getRealException(self):
        if False:
            i = 10
            return i + 15
        return self.real_exception

class ParameterSpec(object):
    __slots__ = ('name', 'owner', 'normal_args', 'normal_variables', 'list_star_arg', 'is_list_star_arg_single', 'dict_star_arg', 'list_star_variable', 'dict_star_variable', 'default_count', 'kw_only_args', 'kw_only_variables', 'pos_only_args', 'pos_only_variables', 'type_shape')

    @counted_init
    def __init__(self, ps_name, ps_normal_args, ps_pos_only_args, ps_kw_only_args, ps_list_star_arg, ps_dict_star_arg, ps_default_count, ps_is_list_star_arg_single=False, type_shape=None):
        if False:
            return 10
        if type(ps_normal_args) is str:
            if ps_normal_args == '':
                ps_normal_args = ()
            else:
                ps_normal_args = ps_normal_args.split(',')
        if type(ps_kw_only_args) is str:
            if ps_kw_only_args == '':
                ps_kw_only_args = ()
            else:
                ps_kw_only_args = ps_kw_only_args.split(',')
        assert None not in ps_normal_args
        self.owner = None
        self.name = ps_name
        self.normal_args = tuple(ps_normal_args)
        self.normal_variables = None
        assert ps_list_star_arg is None or type(ps_list_star_arg) is str, ps_list_star_arg
        assert ps_dict_star_arg is None or type(ps_dict_star_arg) is str, ps_dict_star_arg
        assert type(ps_is_list_star_arg_single) is bool, ps_is_list_star_arg_single
        self.list_star_arg = ps_list_star_arg if ps_list_star_arg else None
        self.is_list_star_arg_single = ps_is_list_star_arg_single
        self.dict_star_arg = ps_dict_star_arg if ps_dict_star_arg else None
        self.list_star_variable = None
        self.dict_star_variable = None
        self.default_count = ps_default_count
        self.kw_only_args = tuple(ps_kw_only_args)
        self.kw_only_variables = None
        self.pos_only_args = tuple(ps_pos_only_args)
        self.pos_only_variables = None
        self.type_shape = type_shape
    if isCountingInstances():
        __del__ = counted_del()

    def makeClone(self):
        if False:
            i = 10
            return i + 15
        return ParameterSpec(ps_name=self.name, ps_normal_args=self.normal_args, ps_pos_only_args=self.pos_only_args, ps_kw_only_args=self.kw_only_args, ps_list_star_arg=self.list_star_arg, ps_dict_star_arg=self.dict_star_arg, ps_default_count=self.default_count, type_shape=self.type_shape)

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'ps_name': self.name, 'ps_normal_args': ','.join(self.normal_args), 'ps_pos_only_args': self.pos_only_args, 'ps_kw_only_args': ','.join(self.kw_only_args), 'ps_list_star_arg': self.list_star_arg if self.list_star_arg is not None else '', 'ps_dict_star_arg': self.dict_star_arg if self.dict_star_arg is not None else '', 'ps_default_count': self.default_count, 'type_shape': self.type_shape}

    def checkParametersValid(self):
        if False:
            return 10
        arg_names = self.getParameterNames()
        for arg_name in arg_names:
            if arg_names.count(arg_name) != 1:
                return "duplicate argument '%s' in function definition" % arg_name
        return None

    def __repr__(self):
        if False:
            print('Hello World!')
        parts = [str(normal_arg) for normal_arg in self.pos_only_args]
        if parts:
            parts.append('/')
        parts += [str(normal_arg) for normal_arg in self.normal_args]
        if self.list_star_arg is not None:
            parts.append('*%s' % self.list_star_arg)
        if self.dict_star_variable is not None:
            parts.append('**%s' % self.dict_star_variable)
        if parts:
            return "<ParameterSpec '%s'>" % ','.join(parts)
        else:
            return '<NoParameters>'

    def setOwner(self, owner):
        if False:
            return 10
        if self.owner is not None:
            return
        self.owner = owner
        self.normal_variables = []
        for normal_arg in self.normal_args:
            if type(normal_arg) is str:
                normal_variable = Variables.ParameterVariable(owner=self.owner, parameter_name=normal_arg)
            else:
                assert False, normal_arg
            self.normal_variables.append(normal_variable)
        if self.list_star_arg:
            self.list_star_variable = Variables.ParameterVariable(owner=owner, parameter_name=self.list_star_arg)
        else:
            self.list_star_variable = None
        if self.dict_star_arg:
            self.dict_star_variable = Variables.ParameterVariable(owner=owner, parameter_name=self.dict_star_arg)
        else:
            self.dict_star_variable = None
        self.kw_only_variables = [Variables.ParameterVariable(owner=self.owner, parameter_name=kw_only_arg) for kw_only_arg in self.kw_only_args]
        self.pos_only_variables = [Variables.ParameterVariable(owner=self.owner, parameter_name=pos_only_arg) for pos_only_arg in self.pos_only_args]

    def getDefaultCount(self):
        if False:
            print('Hello World!')
        return self.default_count

    def hasDefaultParameters(self):
        if False:
            while True:
                i = 10
        return self.getDefaultCount() > 0

    def getTopLevelVariables(self):
        if False:
            for i in range(10):
                print('nop')
        return self.pos_only_variables + self.normal_variables + self.kw_only_variables

    def getAllVariables(self):
        if False:
            while True:
                i = 10
        result = list(self.pos_only_variables)
        result += self.normal_variables
        result += self.kw_only_variables
        if self.list_star_variable is not None:
            result.append(self.list_star_variable)
        if self.dict_star_variable is not None:
            result.append(self.dict_star_variable)
        return result

    def getParameterNames(self):
        if False:
            while True:
                i = 10
        result = list(self.pos_only_args + self.normal_args)
        result += self.kw_only_args
        if self.list_star_arg is not None:
            result.append(self.list_star_arg)
        if self.dict_star_arg is not None:
            result.append(self.dict_star_arg)
        return tuple(result)

    def getStarListArgumentName(self):
        if False:
            return 10
        return self.list_star_arg

    def isStarListSingleArg(self):
        if False:
            i = 10
            return i + 15
        return self.is_list_star_arg_single

    def getListStarArgVariable(self):
        if False:
            return 10
        return self.list_star_variable

    def getStarDictArgumentName(self):
        if False:
            print('Hello World!')
        return self.dict_star_arg

    def getDictStarArgVariable(self):
        if False:
            print('Hello World!')
        return self.dict_star_variable

    def getKwOnlyVariables(self):
        if False:
            print('Hello World!')
        return self.kw_only_variables

    def allowsKeywords(self):
        if False:
            while True:
                i = 10
        return True

    def getKeywordRefusalText(self):
        if False:
            i = 10
            return i + 15
        return '%s() takes no keyword arguments' % self.name

    def getArgumentNames(self):
        if False:
            for i in range(10):
                print('nop')
        return self.pos_only_args + self.normal_args

    def getArgumentCount(self):
        if False:
            i = 10
            return i + 15
        return len(self.normal_args) + len(self.pos_only_args)

    def getKwOnlyParameterNames(self):
        if False:
            i = 10
            return i + 15
        return self.kw_only_args

    def getKwOnlyParameterCount(self):
        if False:
            print('Hello World!')
        return len(self.kw_only_args)

    def getPosOnlyParameterCount(self):
        if False:
            i = 10
            return i + 15
        return len(self.pos_only_args)

    def getTypeShape(self):
        if False:
            return 10
        return self.type_shape

def matchCall(func_name, args, kw_only_args, star_list_arg, star_list_single_arg, star_dict_arg, num_defaults, num_pos_only, positional, pairs, improved=False):
    if False:
        i = 10
        return i + 15
    'Match a call arguments to a signature.\n\n    Args:\n        func_name - Name of the function being matched, used to construct exception texts.\n        args - normal argument names\n        kw_only_args -  keyword only argument names (Python3)\n        star_list_arg - name of star list argument if any\n        star_dict_arg - name of star dict argument if any\n        num_defaults - amount of arguments that have default values\n        num_pos_only - amount of arguments that must be given by position\n        positional - tuple of argument values given for simulated call\n        pairs - tuple of pairs arg argument name and argument values\n        improved - (bool) should we give better errors than CPython or not.\n    Returns:\n        Dictionary of argument name to value mappings\n    Notes:\n        Based loosely on "inspect.getcallargs" with corrections.\n    '
    assert type(positional) is tuple, positional
    assert type(pairs) in (tuple, list), pairs
    pairs = list(pairs)
    result = {}
    assigned_tuple_params = []

    def assign(arg, value):
        if False:
            for i in range(10):
                print('nop')
        if type(arg) is str:
            result[arg] = value
        else:
            assigned_tuple_params.append(arg)
            value = iter(value.getIterationValues())
            for (i, subarg) in enumerate(arg):
                try:
                    subvalue = next(value)
                except StopIteration:
                    raise TooManyArguments(ValueError('need more than %d %s to unpack' % (i, 'values' if i > 1 else 'value')))
                assign(subarg, subvalue)
            try:
                next(value)
            except StopIteration:
                pass
            else:
                raise TooManyArguments(ValueError('too many values to unpack'))

    def isAssigned(arg):
        if False:
            i = 10
            return i + 15
        if type(arg) is str:
            return arg in result
        return arg in assigned_tuple_params
    num_pos = len(positional)
    num_total = num_pos + len(pairs)
    num_args = len(args)
    for (arg, value) in zip(args, positional):
        assign(arg, value)
    if python_version >= 768 and (not star_dict_arg):
        for pair in pairs:
            try:
                arg_index = (args + kw_only_args).index(pair[0])
            except ValueError:
                if improved or python_version >= 880:
                    message = "'%s' is an invalid keyword argument for %s()" % (pair[0], func_name)
                else:
                    message = "'%s' is an invalid keyword argument for this function" % pair[0]
                raise TooManyArguments(TypeError(message))
            if arg_index < num_pos_only:
                message = "'%s' is an invalid keyword argument for %s()" % (pair[0], func_name)
                raise TooManyArguments(TypeError(message))
    if star_list_arg:
        if num_pos > num_args:
            value = positional[-(num_pos - num_args):]
            assign(star_list_arg, value)
            if star_list_single_arg:
                if len(value) > 1:
                    raise TooManyArguments(TypeError('%s expected at most 1 arguments, got %d' % (func_name, len(value))))
        else:
            assign(star_list_arg, ())
    elif 0 < num_args < num_total:
        if num_defaults == 0:
            if num_args == 1:
                raise TooManyArguments(TypeError('%s() takes exactly one argument (%d given)' % (func_name, num_total)))
            raise TooManyArguments(TypeError('%s expected %d arguments, got %d' % (func_name, num_args, num_total)))
        raise TooManyArguments(TypeError('%s() takes at most %d %s (%d given)' % (func_name, num_args, 'argument' if num_args == 1 else 'arguments', num_total)))
    elif num_args == 0 and num_total:
        if star_dict_arg:
            if num_pos:
                raise TooManyArguments(TypeError('%s() takes exactly 0 arguments (%d given)' % (func_name, num_total)))
        else:
            raise TooManyArguments(TypeError('%s() takes no arguments (%d given)' % (func_name, num_total)))
    named_argument_names = [pair[0] for pair in pairs]
    for arg in args + kw_only_args:
        if type(arg) is str and arg in named_argument_names:
            if isAssigned(arg):
                raise TooManyArguments(TypeError("%s() got multiple values for keyword argument '%s'" % (func_name, arg)))
            new_pairs = []
            for pair in pairs:
                if arg == pair[0]:
                    assign(arg, pair[1])
                else:
                    new_pairs.append(pair)
            assert len(new_pairs) == len(pairs) - 1
            pairs = new_pairs
    if num_defaults > 0:
        for arg in (args + kw_only_args)[-num_defaults:]:
            if not isAssigned(arg):
                assign(arg, None)
    if star_dict_arg:
        assign(star_dict_arg, pairs)
    elif pairs:
        unexpected = next(iter(dict(pairs)))
        if improved:
            message = "%s() got an unexpected keyword argument '%s'" % (func_name, unexpected)
        else:
            message = "'%s' is an invalid keyword argument for this function" % unexpected
        raise TooManyArguments(TypeError(message))
    unassigned = num_args - len([arg for arg in args if isAssigned(arg)])
    if unassigned:
        num_required = num_args - num_defaults
        if num_required > 0 or improved:
            if num_defaults == 0 and num_args != 1:
                raise TooManyArguments(TypeError('%s expected %d arguments, got %d' % (func_name, num_args, num_total)))
            if num_required == 1:
                arg_desc = '1 argument' if python_version < 848 else 'one argument'
            else:
                arg_desc = '%d arguments' % num_required
            raise TooManyArguments(TypeError('%s() takes %s %s (%d given)' % (func_name, 'at least' if num_defaults > 0 else 'exactly', arg_desc, num_total)))
        raise TooManyArguments(TypeError('%s expected %s%s, got %d' % (func_name, ('at least ' if python_version < 768 else '') if num_defaults > 0 else 'exactly ', '%d arguments' % num_required, num_total)))
    unassigned = len(kw_only_args) - len([arg for arg in kw_only_args if isAssigned(arg)])
    if unassigned:
        raise TooManyArguments(TypeError('%s missing %d required keyword-only argument%s: %s' % (func_name, unassigned, 's' if unassigned > 1 else '', ' and '.join("'%s'" % [arg for arg in kw_only_args if not isAssigned(arg)]))))
    return result