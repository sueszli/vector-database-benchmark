import itertools
import logging
import typing as T
from . import enums, errors, utils
logger = logging.getLogger('factory.generate')

class BaseDeclaration(utils.OrderedBase):
    """A factory declaration.

    Declarations mark an attribute as needing lazy evaluation.
    This allows them to refer to attributes defined by other BaseDeclarations
    in the same factory.
    """
    FACTORY_BUILDER_PHASE = enums.BuilderPhase.ATTRIBUTE_RESOLUTION
    CAPTURE_OVERRIDES = False
    UNROLL_CONTEXT_BEFORE_EVALUATION = True

    def __init__(self, **defaults):
        if False:
            return 10
        super().__init__()
        self._defaults = defaults or {}

    def unroll_context(self, instance, step, context):
        if False:
            i = 10
            return i + 15
        full_context = dict()
        full_context.update(self._defaults)
        full_context.update(context)
        if not self.UNROLL_CONTEXT_BEFORE_EVALUATION:
            return full_context
        if not any((enums.get_builder_phase(v) for v in full_context.values())):
            return full_context
        import factory.base
        subfactory = factory.base.DictFactory
        return step.recurse(subfactory, full_context, force_sequence=step.sequence)

    def _unwrap_evaluate_pre(self, wrapped, *, instance, step, overrides):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate a wrapped pre-declaration.\n\n        This is especially useful for declarations wrapping another one,\n        e.g. Maybe or Transformer.\n        '
        if isinstance(wrapped, BaseDeclaration):
            return wrapped.evaluate_pre(instance=instance, step=step, overrides=overrides)
        return wrapped

    def evaluate_pre(self, instance, step, overrides):
        if False:
            i = 10
            return i + 15
        context = self.unroll_context(instance, step, overrides)
        return self.evaluate(instance, step, context)

    def evaluate(self, instance, step, extra):
        if False:
            i = 10
            return i + 15
        'Evaluate this declaration.\n\n        Args:\n            instance (builder.Resolver): The object holding currently computed\n                attributes\n            step: a factory.builder.BuildStep\n            extra (dict): additional, call-time added kwargs\n                for the step.\n        '
        raise NotImplementedError('This is an abstract method')

class OrderedDeclaration(BaseDeclaration):
    """Compatibility"""

class LazyFunction(BaseDeclaration):
    """Simplest BaseDeclaration computed by calling the given function.

    Attributes:
        function (function): a function without arguments and
            returning the computed value.
    """

    def __init__(self, function):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.function = function

    def evaluate(self, instance, step, extra):
        if False:
            while True:
                i = 10
        logger.debug('LazyFunction: Evaluating %r on %r', self.function, step)
        return self.function()

class LazyAttribute(BaseDeclaration):
    """Specific BaseDeclaration computed using a lambda.

    Attributes:
        function (function): a function, expecting the current LazyStub and
            returning the computed value.
    """

    def __init__(self, function):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.function = function

    def evaluate(self, instance, step, extra):
        if False:
            i = 10
            return i + 15
        logger.debug('LazyAttribute: Evaluating %r on %r', self.function, instance)
        return self.function(instance)

class Transformer(BaseDeclaration):
    CAPTURE_OVERRIDES = True
    UNROLL_CONTEXT_BEFORE_EVALUATION = False

    class Force:
        """
        Bypass a transformer's transformation.

        The forced value can be any declaration, and will be evaluated as if it
        had been passed instead of the Transformer declaration.
        """

        def __init__(self, forced_value):
            if False:
                while True:
                    i = 10
            self.forced_value = forced_value

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return f'Transformer.Force({repr(self.forced_value)})'

    def __init__(self, default, *, transform):
        if False:
            return 10
        super().__init__()
        self.default = default
        self.transform = transform

    def evaluate_pre(self, instance, step, overrides):
        if False:
            while True:
                i = 10
        value_or_declaration = overrides.pop('', self.default)
        if isinstance(value_or_declaration, self.Force):
            bypass_transform = True
            value_or_declaration = value_or_declaration.forced_value
        else:
            bypass_transform = False
        value = self._unwrap_evaluate_pre(value_or_declaration, instance=instance, step=step, overrides=overrides)
        if bypass_transform:
            return value
        return self.transform(value)

class _UNSPECIFIED:
    pass

def deepgetattr(obj, name, default=_UNSPECIFIED):
    if False:
        for i in range(10):
            print('nop')
    "Try to retrieve the given attribute of an object, digging on '.'.\n\n    This is an extended getattr, digging deeper if '.' is found.\n\n    Args:\n        obj (object): the object of which an attribute should be read\n        name (str): the name of an attribute to look up.\n        default (object): the default value to use if the attribute wasn't found\n\n    Returns:\n        the attribute pointed to by 'name', splitting on '.'.\n\n    Raises:\n        AttributeError: if obj has no 'name' attribute.\n    "
    try:
        if '.' in name:
            (attr, subname) = name.split('.', 1)
            return deepgetattr(getattr(obj, attr), subname, default)
        else:
            return getattr(obj, name)
    except AttributeError:
        if default is _UNSPECIFIED:
            raise
        else:
            return default

class SelfAttribute(BaseDeclaration):
    """Specific BaseDeclaration copying values from other fields.

    If the field name starts with two dots or more, the lookup will be anchored
    in the related 'parent'.

    Attributes:
        depth (int): the number of steps to go up in the containers chain
        attribute_name (str): the name of the attribute to copy.
        default (object): the default value to use if the attribute doesn't
            exist.
    """

    def __init__(self, attribute_name, default=_UNSPECIFIED):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        depth = len(attribute_name) - len(attribute_name.lstrip('.'))
        attribute_name = attribute_name[depth:]
        self.depth = depth
        self.attribute_name = attribute_name
        self.default = default

    def evaluate(self, instance, step, extra):
        if False:
            for i in range(10):
                print('nop')
        if self.depth > 1:
            target = step.chain[self.depth - 1]
        else:
            target = instance
        logger.debug('SelfAttribute: Picking attribute %r on %r', self.attribute_name, target)
        return deepgetattr(target, self.attribute_name, self.default)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s(%r, default=%r)>' % (self.__class__.__name__, self.attribute_name, self.default)

class Iterator(BaseDeclaration):
    """Fill this value using the values returned by an iterator.

    Warning: the iterator should not end !

    Attributes:
        iterator (iterable): the iterator whose value should be used.
        getter (callable or None): a function to parse returned values
    """

    def __init__(self, iterator, cycle=True, getter=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.getter = getter
        self.iterator = None
        if cycle:
            self.iterator_builder = lambda : utils.ResetableIterator(itertools.cycle(iterator))
        else:
            self.iterator_builder = lambda : utils.ResetableIterator(iterator)

    def evaluate(self, instance, step, extra):
        if False:
            for i in range(10):
                print('nop')
        if self.iterator is None:
            self.iterator = self.iterator_builder()
        logger.debug('Iterator: Fetching next value from %r', self.iterator)
        value = next(iter(self.iterator))
        if self.getter is None:
            return value
        return self.getter(value)

    def reset(self):
        if False:
            while True:
                i = 10
        'Reset the internal iterator.'
        if self.iterator is not None:
            self.iterator.reset()

class Sequence(BaseDeclaration):
    """Specific BaseDeclaration to use for 'sequenced' fields.

    These fields are typically used to generate increasing unique values.

    Attributes:
        function (function): A function, expecting the current sequence counter
            and returning the computed value.
    """

    def __init__(self, function):
        if False:
            while True:
                i = 10
        super().__init__()
        self.function = function

    def evaluate(self, instance, step, extra):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Sequence: Computing next value of %r for seq=%s', self.function, step.sequence)
        return self.function(int(step.sequence))

class LazyAttributeSequence(Sequence):
    """Composite of a LazyAttribute and a Sequence.

    Attributes:
        function (function): A function, expecting the current LazyStub and the
            current sequence counter.
        type (function): A function converting an integer into the expected kind
            of counter for the 'function' attribute.
    """

    def evaluate(self, instance, step, extra):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('LazyAttributeSequence: Computing next value of %r for seq=%s, obj=%r', self.function, step.sequence, instance)
        return self.function(instance, int(step.sequence))

class ContainerAttribute(BaseDeclaration):
    """Variant of LazyAttribute, also receives the containers of the object.

    Attributes:
        function (function): A function, expecting the current LazyStub and the
            (optional) object having a subfactory containing this attribute.
        strict (bool): Whether evaluating should fail when the containers are
            not passed in (i.e used outside a SubFactory).
    """

    def __init__(self, function, strict=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.function = function
        self.strict = strict

    def evaluate(self, instance, step, extra):
        if False:
            print('Hello World!')
        'Evaluate the current ContainerAttribute.\n\n        Args:\n            obj (LazyStub): a lazy stub of the object being constructed, if\n                needed.\n            containers (list of LazyStub): a list of lazy stubs of factories\n                being evaluated in a chain, each item being a future field of\n                next one.\n        '
        chain = step.chain[1:]
        if self.strict and (not chain):
            raise TypeError("A ContainerAttribute in 'strict' mode can only be used within a SubFactory.")
        return self.function(instance, chain)

class ParameteredAttribute(BaseDeclaration):
    """Base class for attributes expecting parameters.

    Attributes:
        defaults (dict): Default values for the parameters.
            May be overridden by call-time parameters.
    """

    def evaluate(self, instance, step, extra):
        if False:
            while True:
                i = 10
        'Evaluate the current definition and fill its attributes.\n\n        Uses attributes definition in the following order:\n        - values defined when defining the ParameteredAttribute\n        - additional values defined when instantiating the containing factory\n\n        Args:\n            instance (builder.Resolver): The object holding currently computed\n                attributes\n            step: a factory.builder.BuildStep\n            extra (dict): additional, call-time added kwargs\n                for the step.\n        '
        return self.generate(step, extra)

    def generate(self, step, params):
        if False:
            while True:
                i = 10
        "Actually generate the related attribute.\n\n        Args:\n            sequence (int): the current sequence number\n            obj (LazyStub): the object being constructed\n            create (bool): whether the calling factory was in 'create' or\n                'build' mode\n            params (dict): parameters inherited from init and evaluation-time\n                overrides.\n\n        Returns:\n            Computed value for the current declaration.\n        "
        raise NotImplementedError()

class _FactoryWrapper:
    """Handle a 'factory' arg.

    Such args can be either a Factory subclass, or a fully qualified import
    path for that subclass (e.g 'myapp.factories.MyFactory').
    """

    def __init__(self, factory_or_path):
        if False:
            print('Hello World!')
        self.factory = None
        self.module = self.name = ''
        if isinstance(factory_or_path, type):
            self.factory = factory_or_path
        else:
            if not (isinstance(factory_or_path, str) and '.' in factory_or_path):
                raise ValueError('A factory= argument must receive either a class or the fully qualified path to a Factory subclass; got %r instead.' % factory_or_path)
            (self.module, self.name) = factory_or_path.rsplit('.', 1)

    def get(self):
        if False:
            return 10
        if self.factory is None:
            self.factory = utils.import_object(self.module, self.name)
        return self.factory

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.factory is None:
            return f'<_FactoryImport: {self.module}.{self.name}>'
        else:
            return f'<_FactoryImport: {self.factory.__class__}>'

class SubFactory(BaseDeclaration):
    """Base class for attributes based upon a sub-factory.

    Attributes:
        defaults (dict): Overrides to the defaults defined in the wrapped
            factory
        factory (base.Factory): the wrapped factory
    """
    FORCE_SEQUENCE = False
    UNROLL_CONTEXT_BEFORE_EVALUATION = False

    def __init__(self, factory, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.factory_wrapper = _FactoryWrapper(factory)

    def get_factory(self):
        if False:
            while True:
                i = 10
        'Retrieve the wrapped factory.Factory subclass.'
        return self.factory_wrapper.get()

    def evaluate(self, instance, step, extra):
        if False:
            while True:
                i = 10
        'Evaluate the current definition and fill its attributes.\n\n        Args:\n            step: a factory.builder.BuildStep\n            params (dict): additional, call-time added kwargs\n                for the step.\n        '
        subfactory = self.get_factory()
        logger.debug('SubFactory: Instantiating %s.%s(%s), create=%r', subfactory.__module__, subfactory.__name__, utils.log_pprint(kwargs=extra), step)
        force_sequence = step.sequence if self.FORCE_SEQUENCE else None
        return step.recurse(subfactory, extra, force_sequence=force_sequence)

class Dict(SubFactory):
    """Fill a dict with usual declarations."""
    FORCE_SEQUENCE = True

    def __init__(self, params, dict_factory='factory.DictFactory'):
        if False:
            return 10
        super().__init__(dict_factory, **dict(params))

class List(SubFactory):
    """Fill a list with standard declarations."""
    FORCE_SEQUENCE = True

    def __init__(self, params, list_factory='factory.ListFactory'):
        if False:
            while True:
                i = 10
        params = {str(i): v for (i, v) in enumerate(params)}
        super().__init__(list_factory, **params)

class Skip:

    def __bool__(self):
        if False:
            return 10
        return False
SKIP = Skip()

class Maybe(BaseDeclaration):

    def __init__(self, decider, yes_declaration=SKIP, no_declaration=SKIP):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if enums.get_builder_phase(decider) is None:
            decider = SelfAttribute(decider, default=None)
        self.decider = decider
        self.yes = yes_declaration
        self.no = no_declaration
        phases = {'yes_declaration': enums.get_builder_phase(yes_declaration), 'no_declaration': enums.get_builder_phase(no_declaration)}
        used_phases = {phase for phase in phases.values() if phase is not None}
        if len(used_phases) > 1:
            raise TypeError(f'Inconsistent phases for {self!r}: {phases!r}')
        self.FACTORY_BUILDER_PHASE = used_phases.pop() if used_phases else enums.BuilderPhase.ATTRIBUTE_RESOLUTION

    def evaluate_post(self, instance, step, overrides):
        if False:
            return 10
        'Handle post-generation declarations'
        decider_phase = enums.get_builder_phase(self.decider)
        if decider_phase == enums.BuilderPhase.ATTRIBUTE_RESOLUTION:
            choice = self.decider.evaluate_pre(instance=step.stub, step=step, overrides=overrides)
        else:
            assert decider_phase == enums.BuilderPhase.POST_INSTANTIATION
            choice = self.decider.evaluate_post(instance=instance, step=step, overrides={})
        target = self.yes if choice else self.no
        if enums.get_builder_phase(target) == enums.BuilderPhase.POST_INSTANTIATION:
            return target.evaluate_post(instance=instance, step=step, overrides=overrides)
        else:
            return target

    def evaluate_pre(self, instance, step, overrides):
        if False:
            return 10
        choice = self.decider.evaluate(instance=instance, step=step, extra={})
        target = self.yes if choice else self.no
        return self._unwrap_evaluate_pre(target, instance=instance, step=step, overrides=overrides)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'Maybe({self.decider!r}, yes={self.yes!r}, no={self.no!r})'

class Parameter(utils.OrderedBase):
    """A complex parameter, to be used in a Factory.Params section.

    Must implement:
    - A "compute" function, performing the actual declaration override
    - Optionally, a get_revdeps() function (to compute other parameters it may alter)
    """

    def as_declarations(self, field_name, declarations):
        if False:
            while True:
                i = 10
        'Compute the overrides for this parameter.\n\n        Args:\n        - field_name (str): the field this parameter is installed at\n        - declarations (dict): the global factory declarations\n\n        Returns:\n            dict: the declarations to override\n        '
        raise NotImplementedError()

    def get_revdeps(self, parameters):
        if False:
            while True:
                i = 10
        'Retrieve the list of other parameters modified by this one.'
        return []

class SimpleParameter(Parameter):

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.value = value

    def as_declarations(self, field_name, declarations):
        if False:
            while True:
                i = 10
        return {field_name: self.value}

    @classmethod
    def wrap(cls, value):
        if False:
            return 10
        if not isinstance(value, Parameter):
            return cls(value)
        value.touch_creation_counter()
        return value

class Trait(Parameter):
    """The simplest complex parameter, it enables a bunch of new declarations based on a boolean flag."""

    def __init__(self, **overrides):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.overrides = overrides

    def as_declarations(self, field_name, declarations):
        if False:
            i = 10
            return i + 15
        overrides = {}
        for (maybe_field, new_value) in self.overrides.items():
            overrides[maybe_field] = Maybe(decider=SelfAttribute('%s.%s' % ('.' * maybe_field.count(enums.SPLITTER), field_name), default=False), yes_declaration=new_value, no_declaration=declarations.get(maybe_field, SKIP))
        return overrides

    def get_revdeps(self, parameters):
        if False:
            return 10
        "This might alter fields it's injecting."
        return [param for param in parameters if param in self.overrides]

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%s)' % (self.__class__.__name__, ', '.join(('%s=%r' % t for t in self.overrides.items())))

class PostGenerationContext(T.NamedTuple):
    value_provided: bool
    value: T.Any
    extra: T.Dict[str, T.Any]

class PostGenerationDeclaration(BaseDeclaration):
    """Declarations to be called once the model object has been generated."""
    FACTORY_BUILDER_PHASE = enums.BuilderPhase.POST_INSTANTIATION

    def evaluate_post(self, instance, step, overrides):
        if False:
            i = 10
            return i + 15
        context = self.unroll_context(instance, step, overrides)
        postgen_context = PostGenerationContext(value_provided=bool('' in context), value=context.get(''), extra={k: v for (k, v) in context.items() if k != ''})
        return self.call(instance, step, postgen_context)

    def call(self, instance, step, context):
        if False:
            i = 10
            return i + 15
        "Call this hook; no return value is expected.\n\n        Args:\n            instance (object): the newly generated object\n            step (bool): whether the object was 'built' or 'created'\n            context: a builder.PostGenerationContext containing values\n                extracted from the containing factory's declaration\n        "
        raise NotImplementedError()

class PostGeneration(PostGenerationDeclaration):
    """Calls a given function once the object has been generated."""

    def __init__(self, function):
        if False:
            return 10
        super().__init__()
        self.function = function

    def call(self, instance, step, context):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('PostGeneration: Calling %s.%s(%s)', self.function.__module__, self.function.__name__, utils.log_pprint((instance, step), context._asdict()))
        create = step.builder.strategy == enums.CREATE_STRATEGY
        return self.function(instance, create, context.value, **context.extra)

class RelatedFactory(PostGenerationDeclaration):
    """Calls a factory once the object has been generated.

    Attributes:
        factory (Factory): the factory to call
        defaults (dict): extra declarations for calling the related factory
        name (str): the name to use to refer to the generated object when
            calling the related factory
    """
    UNROLL_CONTEXT_BEFORE_EVALUATION = False

    def __init__(self, factory, factory_related_name='', **defaults):
        if False:
            print('Hello World!')
        super().__init__()
        self.name = factory_related_name
        self.defaults = defaults
        self.factory_wrapper = _FactoryWrapper(factory)

    def get_factory(self):
        if False:
            print('Hello World!')
        'Retrieve the wrapped factory.Factory subclass.'
        return self.factory_wrapper.get()

    def call(self, instance, step, context):
        if False:
            while True:
                i = 10
        factory = self.get_factory()
        if context.value_provided:
            logger.debug('RelatedFactory: Using provided %r instead of generating %s.%s.', context.value, factory.__module__, factory.__name__)
            return context.value
        passed_kwargs = dict(self.defaults)
        passed_kwargs.update(context.extra)
        if self.name:
            passed_kwargs[self.name] = instance
        logger.debug('RelatedFactory: Generating %s.%s(%s)', factory.__module__, factory.__name__, utils.log_pprint((step,), passed_kwargs))
        return step.recurse(factory, passed_kwargs)

class RelatedFactoryList(RelatedFactory):
    """Calls a factory 'size' times once the object has been generated.

    Attributes:
        factory (Factory): the factory to call "size-times"
        defaults (dict): extra declarations for calling the related factory
        factory_related_name (str): the name to use to refer to the generated
            object when calling the related factory
        size (int|lambda): the number of times 'factory' is called, ultimately
            returning a list of 'factory' objects w/ size 'size'.
    """

    def __init__(self, factory, factory_related_name='', size=2, **defaults):
        if False:
            for i in range(10):
                print('nop')
        self.size = size
        super().__init__(factory, factory_related_name, **defaults)

    def call(self, instance, step, context):
        if False:
            i = 10
            return i + 15
        parent = super()
        return [parent.call(instance, step, context) for i in range(self.size if isinstance(self.size, int) else self.size())]

class NotProvided:
    pass

class PostGenerationMethodCall(PostGenerationDeclaration):
    """Calls a method of the generated object.

    Attributes:
        method_name (str): the method to call
        method_args (list): arguments to pass to the method
        method_kwargs (dict): keyword arguments to pass to the method

    Example:
        class UserFactory(factory.Factory):
            ...
            password = factory.PostGenerationMethodCall('set_pass', password='')
    """

    def __init__(self, method_name, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        if len(args) > 1:
            raise errors.InvalidDeclarationError('A PostGenerationMethodCall can only handle 1 positional argument; please provide other parameters through keyword arguments.')
        self.method_name = method_name
        self.method_arg = args[0] if args else NotProvided
        self.method_kwargs = kwargs

    def call(self, instance, step, context):
        if False:
            return 10
        if not context.value_provided:
            if self.method_arg is NotProvided:
                args = ()
            else:
                args = (self.method_arg,)
        else:
            args = (context.value,)
        kwargs = dict(self.method_kwargs)
        kwargs.update(context.extra)
        method = getattr(instance, self.method_name)
        logger.debug('PostGenerationMethodCall: Calling %r.%s(%s)', instance, self.method_name, utils.log_pprint(args, kwargs))
        return method(*args, **kwargs)