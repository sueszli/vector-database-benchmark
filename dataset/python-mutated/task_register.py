"""
Define the centralized register of all :class:`~luigi.task.Task` classes.
"""
import abc
import logging
logger = logging.getLogger('luigi-interface')

class TaskClassException(Exception):
    pass

class TaskClassNotFoundException(TaskClassException):
    pass

class TaskClassAmbigiousException(TaskClassException):
    pass

class Register(abc.ABCMeta):
    """
    The Metaclass of :py:class:`Task`.

    Acts as a global registry of Tasks with the following properties:

    1. Cache instances of objects so that eg. ``X(1, 2, 3)`` always returns the
       same object.
    2. Keep track of all subclasses of :py:class:`Task` and expose them.
    """
    __instance_cache = {}
    _default_namespace_dict = {}
    _reg = []
    AMBIGUOUS_CLASS = object()
    'If this value is returned by :py:meth:`_get_reg` then there is an\n    ambiguous task name (two :py:class:`Task` have the same name). This denotes\n    an error.'

    def __new__(metacls, classname, bases, classdict, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Custom class creation for namespacing.\n\n        Also register all subclasses.\n\n        When the set or inherited namespace evaluates to ``None``, set the task namespace to\n        whatever the currently declared namespace is.\n        '
        cls = super(Register, metacls).__new__(metacls, classname, bases, classdict, **kwargs)
        cls._namespace_at_class_time = metacls._get_namespace(cls.__module__)
        metacls._reg.append(cls)
        return cls

    def __call__(cls, *args, **kwargs):
        if False:
            return 10
        '\n        Custom class instantiation utilizing instance cache.\n\n        If a Task has already been instantiated with the same parameters,\n        the previous instance is returned to reduce number of object instances.\n        '

        def instantiate():
            if False:
                while True:
                    i = 10
            return super(Register, cls).__call__(*args, **kwargs)
        h = cls.__instance_cache
        if h is None:
            return instantiate()
        params = cls.get_params()
        param_values = cls.get_param_values(params, args, kwargs)
        k = (cls, tuple(param_values))
        try:
            hash(k)
        except TypeError:
            logger.debug("Not all parameter values are hashable so instance isn't coming from the cache")
            return instantiate()
        if k not in h:
            h[k] = instantiate()
        return h[k]

    @classmethod
    def clear_instance_cache(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear/Reset the instance cache.\n        '
        cls.__instance_cache = {}

    @classmethod
    def disable_instance_cache(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Disables the instance cache.\n        '
        cls.__instance_cache = None

    @property
    def task_family(cls):
        if False:
            print('Hello World!')
        '\n        Internal note: This function will be deleted soon.\n        '
        task_namespace = cls.get_task_namespace()
        if not task_namespace:
            return cls.__name__
        else:
            return f'{task_namespace}.{cls.__name__}'

    @classmethod
    def _get_reg(cls):
        if False:
            i = 10
            return i + 15
        'Return all of the registered classes.\n\n        :return:  an ``dict`` of task_family -> class\n        '
        reg = dict()
        for task_cls in cls._reg:
            if not task_cls._visible_in_registry:
                continue
            name = task_cls.get_task_family()
            if name in reg and (reg[name] == Register.AMBIGUOUS_CLASS or not issubclass(task_cls, reg[name])):
                reg[name] = Register.AMBIGUOUS_CLASS
            else:
                reg[name] = task_cls
        return reg

    @classmethod
    def _set_reg(cls, reg):
        if False:
            i = 10
            return i + 15
        'The writing complement of _get_reg\n        '
        cls._reg = [task_cls for task_cls in reg.values() if task_cls is not cls.AMBIGUOUS_CLASS]

    @classmethod
    def task_names(cls):
        if False:
            print('Hello World!')
        '\n        List of task names as strings\n        '
        return sorted(cls._get_reg().keys())

    @classmethod
    def tasks_str(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Human-readable register contents dump.\n        '
        return ','.join(cls.task_names())

    @classmethod
    def get_task_cls(cls, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an unambiguous class or raises an exception.\n        '
        task_cls = cls._get_reg().get(name)
        if not task_cls:
            raise TaskClassNotFoundException(cls._missing_task_msg(name))
        if task_cls == cls.AMBIGUOUS_CLASS:
            raise TaskClassAmbigiousException('Task %r is ambiguous' % name)
        return task_cls

    @classmethod
    def get_all_params(cls):
        if False:
            i = 10
            return i + 15
        '\n        Compiles and returns all parameters for all :py:class:`Task`.\n\n        :return: a generator of tuples (TODO: we should make this more elegant)\n        '
        for (task_name, task_cls) in cls._get_reg().items():
            if task_cls == cls.AMBIGUOUS_CLASS:
                continue
            for (param_name, param_obj) in task_cls.get_params():
                yield (task_name, not task_cls.use_cmdline_section, param_name, param_obj)

    @staticmethod
    def _editdistance(a, b):
        if False:
            for i in range(10):
                print('nop')
        ' Simple unweighted Levenshtein distance '
        r0 = range(0, len(b) + 1)
        r1 = [0] * (len(b) + 1)
        for i in range(0, len(a)):
            r1[0] = i + 1
            for j in range(0, len(b)):
                c = 0 if a[i] is b[j] else 1
                r1[j + 1] = min(r1[j] + 1, r0[j + 1] + 1, r0[j] + c)
            r0 = r1[:]
        return r1[len(b)]

    @classmethod
    def _missing_task_msg(cls, task_name):
        if False:
            i = 10
            return i + 15
        weighted_tasks = [(Register._editdistance(task_name, task_name_2), task_name_2) for task_name_2 in cls.task_names()]
        ordered_tasks = sorted(weighted_tasks, key=lambda pair: pair[0])
        candidates = [task for (dist, task) in ordered_tasks if dist <= 5 and dist < len(task)]
        if candidates:
            return 'No task %s. Did you mean:\n%s' % (task_name, '\n'.join(candidates))
        else:
            return 'No task %s. Candidates are: %s' % (task_name, cls.tasks_str())

    @classmethod
    def _get_namespace(mcs, module_name):
        if False:
            i = 10
            return i + 15
        for parent in mcs._module_parents(module_name):
            entry = mcs._default_namespace_dict.get(parent)
            if entry:
                return entry
        return ''

    @staticmethod
    def _module_parents(module_name):
        if False:
            for i in range(10):
                print('nop')
        "\n        >>> list(Register._module_parents('a.b'))\n        ['a.b', 'a', '']\n        "
        spl = module_name.split('.')
        for i in range(len(spl), 0, -1):
            yield '.'.join(spl[0:i])
        if module_name:
            yield ''

def load_task(module, task_name, params_str):
    if False:
        while True:
            i = 10
    '\n    Imports task dynamically given a module and a task name.\n    '
    if module is not None:
        __import__(module)
    task_cls = Register.get_task_cls(task_name)
    return task_cls.from_str_params(params_str)