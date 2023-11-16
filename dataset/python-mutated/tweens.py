from zope.interface import implementer
from pyramid.config.actions import action_method
from pyramid.exceptions import ConfigurationError
from pyramid.interfaces import ITweens
from pyramid.tweens import EXCVIEW, INGRESS, MAIN
from pyramid.util import TopologicalSorter, is_nonstr_iter, is_string_or_iterable

class TweensConfiguratorMixin:

    def add_tween(self, tween_factory, under=None, over=None):
        if False:
            print('Hello World!')
        "\n        .. versionadded:: 1.2\n\n        Add a 'tween factory'.  A :term:`tween` (a contraction of 'between')\n        is a bit of code that sits between the Pyramid router's main request\n        handling function and the upstream WSGI component that uses\n        :app:`Pyramid` as its 'app'.  Tweens are a feature that may be used\n        by Pyramid framework extensions, to provide, for example,\n        Pyramid-specific view timing support, bookkeeping code that examines\n        exceptions before they are returned to the upstream WSGI application,\n        or a variety of other features.  Tweens behave a bit like\n        :term:`WSGI` 'middleware' but they have the benefit of running in a\n        context in which they have access to the Pyramid :term:`application\n        registry` as well as the Pyramid rendering machinery.\n\n        .. note:: You can view the tween ordering configured into a given\n                  Pyramid application by using the ``ptweens``\n                  command.  See :ref:`displaying_tweens`.\n\n        The ``tween_factory`` argument must be a :term:`dotted Python name`\n        to a global object representing the tween factory.\n\n        The ``under`` and ``over`` arguments allow the caller of\n        ``add_tween`` to provide a hint about where in the tween chain this\n        tween factory should be placed when an implicit tween chain is used.\n        These hints are only used when an explicit tween chain is not used\n        (when the ``pyramid.tweens`` configuration value is not set).\n        Allowable values for ``under`` or ``over`` (or both) are:\n\n        - ``None`` (the default).\n\n        - A :term:`dotted Python name` to a tween factory: a string\n          representing the dotted name of a tween factory added in a call to\n          ``add_tween`` in the same configuration session.\n\n        - One of the constants :attr:`pyramid.tweens.MAIN`,\n          :attr:`pyramid.tweens.INGRESS`, or :attr:`pyramid.tweens.EXCVIEW`.\n\n        - An iterable of any combination of the above. This allows the user\n          to specify fallbacks if the desired tween is not included, as well\n          as compatibility with multiple other tweens.\n\n        ``under`` means 'closer to the main Pyramid application than',\n        ``over`` means 'closer to the request ingress than'.\n\n        For example, calling ``add_tween('myapp.tfactory',\n        over=pyramid.tweens.MAIN)`` will attempt to place the tween factory\n        represented by the dotted name ``myapp.tfactory`` directly 'above'\n        (in ``ptweens`` order) the main Pyramid request handler.\n        Likewise, calling ``add_tween('myapp.tfactory',\n        over=pyramid.tweens.MAIN, under='mypkg.someothertween')`` will\n        attempt to place this tween factory 'above' the main handler but\n        'below' (a fictional) 'mypkg.someothertween' tween factory.\n\n        If all options for ``under`` (or ``over``) cannot be found in the\n        current configuration, it is an error. If some options are specified\n        purely for compatibility with other tweens, just add a fallback of\n        MAIN or INGRESS. For example, ``under=('mypkg.someothertween',\n        'mypkg.someothertween2', INGRESS)``.  This constraint will require\n        the tween to be located under both the 'mypkg.someothertween' tween,\n        the 'mypkg.someothertween2' tween, and INGRESS. If any of these is\n        not in the current configuration, this constraint will only organize\n        itself based on the tweens that are present.\n\n        Specifying neither ``over`` nor ``under`` is equivalent to specifying\n        ``under=INGRESS``.\n\n        Implicit tween ordering is obviously only best-effort.  Pyramid will\n        attempt to present an implicit order of tweens as best it can, but\n        the only surefire way to get any particular ordering is to use an\n        explicit tween order.  A user may always override the implicit tween\n        ordering by using an explicit ``pyramid.tweens`` configuration value\n        setting.\n\n        ``under``, and ``over`` arguments are ignored when an explicit tween\n        chain is specified using the ``pyramid.tweens`` configuration value.\n\n        For more information, see :ref:`registering_tweens`.\n\n        "
        return self._add_tween(tween_factory, under=under, over=over, explicit=False)

    def add_default_tweens(self):
        if False:
            print('Hello World!')
        self.add_tween(EXCVIEW)

    @action_method
    def _add_tween(self, tween_factory, under=None, over=None, explicit=False):
        if False:
            while True:
                i = 10
        if not isinstance(tween_factory, str):
            raise ConfigurationError('The "tween_factory" argument to add_tween must be a dotted name to a globally importable object, not %r' % tween_factory)
        name = tween_factory
        if name in (MAIN, INGRESS):
            raise ConfigurationError('%s is a reserved tween name' % name)
        tween_factory = self.maybe_dotted(tween_factory)
        for (t, p) in [('over', over), ('under', under)]:
            if p is not None:
                if not is_string_or_iterable(p):
                    raise ConfigurationError(f'"{t}" must be a string or iterable, not {p}')
        if over is INGRESS or (is_nonstr_iter(over) and INGRESS in over):
            raise ConfigurationError('%s cannot be over INGRESS' % name)
        if under is MAIN or (is_nonstr_iter(under) and MAIN in under):
            raise ConfigurationError('%s cannot be under MAIN' % name)
        registry = self.registry
        introspectables = []
        tweens = registry.queryUtility(ITweens)
        if tweens is None:
            tweens = Tweens()
            registry.registerUtility(tweens, ITweens)

        def register():
            if False:
                while True:
                    i = 10
            if explicit:
                tweens.add_explicit(name, tween_factory)
            else:
                tweens.add_implicit(name, tween_factory, under=under, over=over)
        discriminator = ('tween', name, explicit)
        tween_type = explicit and 'explicit' or 'implicit'
        intr = self.introspectable('tweens', discriminator, name, '%s tween' % tween_type)
        intr['name'] = name
        intr['factory'] = tween_factory
        intr['type'] = tween_type
        intr['under'] = under
        intr['over'] = over
        introspectables.append(intr)
        self.action(discriminator, register, introspectables=introspectables)

@implementer(ITweens)
class Tweens:

    def __init__(self):
        if False:
            print('Hello World!')
        self.sorter = TopologicalSorter(default_before=None, default_after=INGRESS, first=INGRESS, last=MAIN)
        self.explicit = []

    def add_explicit(self, name, factory):
        if False:
            return 10
        self.explicit.append((name, factory))

    def add_implicit(self, name, factory, under=None, over=None):
        if False:
            print('Hello World!')
        self.sorter.add(name, factory, after=under, before=over)

    def implicit(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sorter.sorted()

    def __call__(self, handler, registry):
        if False:
            while True:
                i = 10
        if self.explicit:
            use = self.explicit
        else:
            use = self.implicit()
        for (name, factory) in use[::-1]:
            handler = factory(handler, registry)
        return handler