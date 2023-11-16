"""

"""
from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import util as orm_util
from ._typing import insp_is_aliased_class
from ._typing import insp_is_attribute
from ._typing import insp_is_mapper
from ._typing import insp_is_mapper_property
from .attributes import QueryableAttribute
from .base import InspectionAttr
from .interfaces import LoaderOption
from .path_registry import _DEFAULT_TOKEN
from .path_registry import _StrPathToken
from .path_registry import _WILDCARD_TOKEN
from .path_registry import AbstractEntityRegistry
from .path_registry import path_is_property
from .path_registry import PathRegistry
from .path_registry import TokenRegistry
from .util import _orm_full_deannotate
from .util import AliasedInsp
from .. import exc as sa_exc
from .. import inspect
from .. import util
from ..sql import and_
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import traversals
from ..sql import visitors
from ..sql.base import _generative
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Self
_RELATIONSHIP_TOKEN: Final[Literal['relationship']] = 'relationship'
_COLUMN_TOKEN: Final[Literal['column']] = 'column'
_FN = TypeVar('_FN', bound='Callable[..., Any]')
if typing.TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _InternalEntityType
    from .context import _MapperEntity
    from .context import ORMCompileState
    from .context import QueryContext
    from .interfaces import _StrategyKey
    from .interfaces import MapperProperty
    from .interfaces import ORMOption
    from .mapper import Mapper
    from .path_registry import _PathRepresentation
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _FromClauseArgument
    from ..sql.cache_key import _CacheKeyTraversalType
    from ..sql.cache_key import CacheKey
_AttrType = Union[Literal['*'], 'QueryableAttribute[Any]']
_WildcardKeyType = Literal['relationship', 'column']
_StrategySpec = Dict[str, Any]
_OptsType = Dict[str, Any]
_AttrGroupType = Tuple[_AttrType, ...]

class _AbstractLoad(traversals.GenerativeOnTraversal, LoaderOption):
    __slots__ = ('propagate_to_loaders',)
    _is_strategy_option = True
    propagate_to_loaders: bool

    def contains_eager(self, attr: _AttrType, alias: Optional[_FromClauseArgument]=None, _is_chain: bool=False) -> Self:
        if False:
            print('Hello World!')
        "Indicate that the given attribute should be eagerly loaded from\n        columns stated manually in the query.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        The option is used in conjunction with an explicit join that loads\n        the desired rows, i.e.::\n\n            sess.query(Order).\\\n                    join(Order.user).\\\n                    options(contains_eager(Order.user))\n\n        The above query would join from the ``Order`` entity to its related\n        ``User`` entity, and the returned ``Order`` objects would have the\n        ``Order.user`` attribute pre-populated.\n\n        It may also be used for customizing the entries in an eagerly loaded\n        collection; queries will normally want to use the\n        :ref:`orm_queryguide_populate_existing` execution option assuming the\n        primary collection of parent objects may already have been loaded::\n\n            sess.query(User).\\\n                join(User.addresses).\\\n                filter(Address.email_address.like('%@aol.com')).\\\n                options(contains_eager(User.addresses)).\\\n                populate_existing()\n\n        See the section :ref:`contains_eager` for complete usage details.\n\n        .. seealso::\n\n            :ref:`loading_toplevel`\n\n            :ref:`contains_eager`\n\n        "
        if alias is not None:
            if not isinstance(alias, str):
                coerced_alias = coercions.expect(roles.FromClauseRole, alias)
            else:
                util.warn_deprecated("Passing a string name for the 'alias' argument to 'contains_eager()` is deprecated, and will not work in a future release.  Please use a sqlalchemy.alias() or sqlalchemy.orm.aliased() construct.", version='1.4')
                coerced_alias = alias
        elif getattr(attr, '_of_type', None):
            assert isinstance(attr, QueryableAttribute)
            ot: Optional[_InternalEntityType[Any]] = inspect(attr._of_type)
            assert ot is not None
            coerced_alias = ot.selectable
        else:
            coerced_alias = None
        cloned = self._set_relationship_strategy(attr, {'lazy': 'joined'}, propagate_to_loaders=False, opts={'eager_from_alias': coerced_alias}, _reconcile_to_other=True if _is_chain else None)
        return cloned

    def load_only(self, *attrs: _AttrType, raiseload: bool=False) -> Self:
        if False:
            return 10
        'Indicate that for a particular entity, only the given list\n        of column-based attribute names should be loaded; all others will be\n        deferred.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        Example - given a class ``User``, load only the ``name`` and\n        ``fullname`` attributes::\n\n            session.query(User).options(load_only(User.name, User.fullname))\n\n        Example - given a relationship ``User.addresses -> Address``, specify\n        subquery loading for the ``User.addresses`` collection, but on each\n        ``Address`` object load only the ``email_address`` attribute::\n\n            session.query(User).options(\n                subqueryload(User.addresses).load_only(Address.email_address)\n            )\n\n        For a statement that has multiple entities,\n        the lead entity can be\n        specifically referred to using the :class:`_orm.Load` constructor::\n\n            stmt = select(User, Address).join(User.addresses).options(\n                        Load(User).load_only(User.name, User.fullname),\n                        Load(Address).load_only(Address.email_address)\n                    )\n\n        :param \\*attrs: Attributes to be loaded, all others will be deferred.\n\n        :param raiseload: raise :class:`.InvalidRequestError` rather than\n         lazy loading a value when a deferred attribute is accessed. Used\n         to prevent unwanted SQL from being emitted.\n\n         .. versionadded:: 2.0\n\n        .. seealso::\n\n            :ref:`orm_queryguide_column_deferral` - in the\n            :ref:`queryguide_toplevel`\n\n        :param \\*attrs: Attributes to be loaded, all others will be deferred.\n\n        :param raiseload: raise :class:`.InvalidRequestError` rather than\n         lazy loading a value when a deferred attribute is accessed. Used\n         to prevent unwanted SQL from being emitted.\n\n         .. versionadded:: 2.0\n\n        '
        cloned = self._set_column_strategy(attrs, {'deferred': False, 'instrument': True})
        wildcard_strategy = {'deferred': True, 'instrument': True}
        if raiseload:
            wildcard_strategy['raiseload'] = True
        cloned = cloned._set_column_strategy(('*',), wildcard_strategy)
        return cloned

    def joinedload(self, attr: _AttrType, innerjoin: Optional[bool]=None) -> Self:
        if False:
            print('Hello World!')
        'Indicate that the given attribute should be loaded using joined\n        eager loading.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        examples::\n\n            # joined-load the "orders" collection on "User"\n            query(User).options(joinedload(User.orders))\n\n            # joined-load Order.items and then Item.keywords\n            query(Order).options(\n                joinedload(Order.items).joinedload(Item.keywords))\n\n            # lazily load Order.items, but when Items are loaded,\n            # joined-load the keywords collection\n            query(Order).options(\n                lazyload(Order.items).joinedload(Item.keywords))\n\n        :param innerjoin: if ``True``, indicates that the joined eager load\n         should use an inner join instead of the default of left outer join::\n\n            query(Order).options(joinedload(Order.user, innerjoin=True))\n\n        In order to chain multiple eager joins together where some may be\n        OUTER and others INNER, right-nested joins are used to link them::\n\n            query(A).options(\n                joinedload(A.bs, innerjoin=False).\n                    joinedload(B.cs, innerjoin=True)\n            )\n\n        The above query, linking A.bs via "outer" join and B.cs via "inner"\n        join would render the joins as "a LEFT OUTER JOIN (b JOIN c)". When\n        using older versions of SQLite (< 3.7.16), this form of JOIN is\n        translated to use full subqueries as this syntax is otherwise not\n        directly supported.\n\n        The ``innerjoin`` flag can also be stated with the term ``"unnested"``.\n        This indicates that an INNER JOIN should be used, *unless* the join\n        is linked to a LEFT OUTER JOIN to the left, in which case it\n        will render as LEFT OUTER JOIN.  For example, supposing ``A.bs``\n        is an outerjoin::\n\n            query(A).options(\n                joinedload(A.bs).\n                    joinedload(B.cs, innerjoin="unnested")\n            )\n\n        The above join will render as "a LEFT OUTER JOIN b LEFT OUTER JOIN c",\n        rather than as "a LEFT OUTER JOIN (b JOIN c)".\n\n        .. note:: The "unnested" flag does **not** affect the JOIN rendered\n            from a many-to-many association table, e.g. a table configured as\n            :paramref:`_orm.relationship.secondary`, to the target table; for\n            correctness of results, these joins are always INNER and are\n            therefore right-nested if linked to an OUTER join.\n\n        .. note::\n\n            The joins produced by :func:`_orm.joinedload` are **anonymously\n            aliased**. The criteria by which the join proceeds cannot be\n            modified, nor can the ORM-enabled :class:`_sql.Select` or legacy\n            :class:`_query.Query` refer to these joins in any way, including\n            ordering. See :ref:`zen_of_eager_loading` for further detail.\n\n            To produce a specific SQL JOIN which is explicitly available, use\n            :meth:`_sql.Select.join` and :meth:`_query.Query.join`. To combine\n            explicit JOINs with eager loading of collections, use\n            :func:`_orm.contains_eager`; see :ref:`contains_eager`.\n\n        .. seealso::\n\n            :ref:`loading_toplevel`\n\n            :ref:`joined_eager_loading`\n\n        '
        loader = self._set_relationship_strategy(attr, {'lazy': 'joined'}, opts={'innerjoin': innerjoin} if innerjoin is not None else util.EMPTY_DICT)
        return loader

    def subqueryload(self, attr: _AttrType) -> Self:
        if False:
            while True:
                i = 10
        'Indicate that the given attribute should be loaded using\n        subquery eager loading.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        examples::\n\n            # subquery-load the "orders" collection on "User"\n            query(User).options(subqueryload(User.orders))\n\n            # subquery-load Order.items and then Item.keywords\n            query(Order).options(\n                subqueryload(Order.items).subqueryload(Item.keywords))\n\n            # lazily load Order.items, but when Items are loaded,\n            # subquery-load the keywords collection\n            query(Order).options(\n                lazyload(Order.items).subqueryload(Item.keywords))\n\n\n        .. seealso::\n\n            :ref:`loading_toplevel`\n\n            :ref:`subquery_eager_loading`\n\n        '
        return self._set_relationship_strategy(attr, {'lazy': 'subquery'})

    def selectinload(self, attr: _AttrType, recursion_depth: Optional[int]=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Indicate that the given attribute should be loaded using\n        SELECT IN eager loading.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        examples::\n\n            # selectin-load the "orders" collection on "User"\n            query(User).options(selectinload(User.orders))\n\n            # selectin-load Order.items and then Item.keywords\n            query(Order).options(\n                selectinload(Order.items).selectinload(Item.keywords))\n\n            # lazily load Order.items, but when Items are loaded,\n            # selectin-load the keywords collection\n            query(Order).options(\n                lazyload(Order.items).selectinload(Item.keywords))\n\n        :param recursion_depth: optional int; when set to a positive integer\n         in conjunction with a self-referential relationship,\n         indicates "selectin" loading will continue that many levels deep\n         automatically until no items are found.\n\n         .. note:: The :paramref:`_orm.selectinload.recursion_depth` option\n            currently supports only self-referential relationships.  There\n            is not yet an option to automatically traverse recursive structures\n            with more than one relationship involved.\n\n            Additionally, the :paramref:`_orm.selectinload.recursion_depth`\n            parameter is new and experimental and should be treated as "alpha"\n            status for the 2.0 series.\n\n         .. versionadded:: 2.0 added\n            :paramref:`_orm.selectinload.recursion_depth`\n\n\n        .. seealso::\n\n            :ref:`loading_toplevel`\n\n            :ref:`selectin_eager_loading`\n\n        '
        return self._set_relationship_strategy(attr, {'lazy': 'selectin'}, opts={'recursion_depth': recursion_depth})

    def lazyload(self, attr: _AttrType) -> Self:
        if False:
            while True:
                i = 10
        'Indicate that the given attribute should be loaded using "lazy"\n        loading.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        .. seealso::\n\n            :ref:`loading_toplevel`\n\n            :ref:`lazy_loading`\n\n        '
        return self._set_relationship_strategy(attr, {'lazy': 'select'})

    def immediateload(self, attr: _AttrType, recursion_depth: Optional[int]=None) -> Self:
        if False:
            return 10
        'Indicate that the given attribute should be loaded using\n        an immediate load with a per-attribute SELECT statement.\n\n        The load is achieved using the "lazyloader" strategy and does not\n        fire off any additional eager loaders.\n\n        The :func:`.immediateload` option is superseded in general\n        by the :func:`.selectinload` option, which performs the same task\n        more efficiently by emitting a SELECT for all loaded objects.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        :param recursion_depth: optional int; when set to a positive integer\n         in conjunction with a self-referential relationship,\n         indicates "selectin" loading will continue that many levels deep\n         automatically until no items are found.\n\n         .. note:: The :paramref:`_orm.immediateload.recursion_depth` option\n            currently supports only self-referential relationships.  There\n            is not yet an option to automatically traverse recursive structures\n            with more than one relationship involved.\n\n         .. warning:: This parameter is new and experimental and should be\n            treated as "alpha" status\n\n         .. versionadded:: 2.0 added\n            :paramref:`_orm.immediateload.recursion_depth`\n\n\n        .. seealso::\n\n            :ref:`loading_toplevel`\n\n            :ref:`selectin_eager_loading`\n\n        '
        loader = self._set_relationship_strategy(attr, {'lazy': 'immediate'}, opts={'recursion_depth': recursion_depth})
        return loader

    def noload(self, attr: _AttrType) -> Self:
        if False:
            i = 10
            return i + 15
        'Indicate that the given relationship attribute should remain\n        unloaded.\n\n        The relationship attribute will return ``None`` when accessed without\n        producing any loading effect.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        :func:`_orm.noload` applies to :func:`_orm.relationship` attributes\n        only.\n\n        .. note:: Setting this loading strategy as the default strategy\n            for a relationship using the :paramref:`.orm.relationship.lazy`\n            parameter may cause issues with flushes, such if a delete operation\n            needs to load related objects and instead ``None`` was returned.\n\n        .. seealso::\n\n            :ref:`loading_toplevel`\n\n        '
        return self._set_relationship_strategy(attr, {'lazy': 'noload'})

    def raiseload(self, attr: _AttrType, sql_only: bool=False) -> Self:
        if False:
            while True:
                i = 10
        "Indicate that the given attribute should raise an error if accessed.\n\n        A relationship attribute configured with :func:`_orm.raiseload` will\n        raise an :exc:`~sqlalchemy.exc.InvalidRequestError` upon access. The\n        typical way this is useful is when an application is attempting to\n        ensure that all relationship attributes that are accessed in a\n        particular context would have been already loaded via eager loading.\n        Instead of having to read through SQL logs to ensure lazy loads aren't\n        occurring, this strategy will cause them to raise immediately.\n\n        :func:`_orm.raiseload` applies to :func:`_orm.relationship` attributes\n        only. In order to apply raise-on-SQL behavior to a column-based\n        attribute, use the :paramref:`.orm.defer.raiseload` parameter on the\n        :func:`.defer` loader option.\n\n        :param sql_only: if True, raise only if the lazy load would emit SQL,\n         but not if it is only checking the identity map, or determining that\n         the related value should just be None due to missing keys. When False,\n         the strategy will raise for all varieties of relationship loading.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        .. seealso::\n\n            :ref:`loading_toplevel`\n\n            :ref:`prevent_lazy_with_raiseload`\n\n            :ref:`orm_queryguide_deferred_raiseload`\n\n        "
        return self._set_relationship_strategy(attr, {'lazy': 'raise_on_sql' if sql_only else 'raise'})

    def defaultload(self, attr: _AttrType) -> Self:
        if False:
            i = 10
            return i + 15
        'Indicate an attribute should load using its predefined loader style.\n\n        The behavior of this loading option is to not change the current\n        loading style of the attribute, meaning that the previously configured\n        one is used or, if no previous style was selected, the default\n        loading will be used.\n\n        This method is used to link to other loader options further into\n        a chain of attributes without altering the loader style of the links\n        along the chain.  For example, to set joined eager loading for an\n        element of an element::\n\n            session.query(MyClass).options(\n                defaultload(MyClass.someattribute).\n                joinedload(MyOtherClass.someotherattribute)\n            )\n\n        :func:`.defaultload` is also useful for setting column-level options on\n        a related class, namely that of :func:`.defer` and :func:`.undefer`::\n\n            session.query(MyClass).options(\n                defaultload(MyClass.someattribute).\n                defer("some_column").\n                undefer("some_other_column")\n            )\n\n        .. seealso::\n\n            :ref:`orm_queryguide_relationship_sub_options`\n\n            :meth:`_orm.Load.options`\n\n        '
        return self._set_relationship_strategy(attr, None)

    def defer(self, key: _AttrType, raiseload: bool=False) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Indicate that the given column-oriented attribute should be\n        deferred, e.g. not loaded until accessed.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        e.g.::\n\n            from sqlalchemy.orm import defer\n\n            session.query(MyClass).options(\n                defer(MyClass.attribute_one),\n                defer(MyClass.attribute_two)\n            )\n\n        To specify a deferred load of an attribute on a related class,\n        the path can be specified one token at a time, specifying the loading\n        style for each link along the chain.  To leave the loading style\n        for a link unchanged, use :func:`_orm.defaultload`::\n\n            session.query(MyClass).options(\n                defaultload(MyClass.someattr).defer(RelatedClass.some_column)\n            )\n\n        Multiple deferral options related to a relationship can be bundled\n        at once using :meth:`_orm.Load.options`::\n\n\n            session.query(MyClass).options(\n                defaultload(MyClass.someattr).options(\n                    defer(RelatedClass.some_column),\n                    defer(RelatedClass.some_other_column),\n                    defer(RelatedClass.another_column)\n                )\n            )\n\n        :param key: Attribute to be deferred.\n\n        :param raiseload: raise :class:`.InvalidRequestError` rather than\n         lazy loading a value when the deferred attribute is accessed. Used\n         to prevent unwanted SQL from being emitted.\n\n        .. versionadded:: 1.4\n\n        .. seealso::\n\n            :ref:`orm_queryguide_column_deferral` - in the\n            :ref:`queryguide_toplevel`\n\n            :func:`_orm.load_only`\n\n            :func:`_orm.undefer`\n\n        '
        strategy = {'deferred': True, 'instrument': True}
        if raiseload:
            strategy['raiseload'] = True
        return self._set_column_strategy((key,), strategy)

    def undefer(self, key: _AttrType) -> Self:
        if False:
            return 10
        'Indicate that the given column-oriented attribute should be\n        undeferred, e.g. specified within the SELECT statement of the entity\n        as a whole.\n\n        The column being undeferred is typically set up on the mapping as a\n        :func:`.deferred` attribute.\n\n        This function is part of the :class:`_orm.Load` interface and supports\n        both method-chained and standalone operation.\n\n        Examples::\n\n            # undefer two columns\n            session.query(MyClass).options(\n                undefer(MyClass.col1), undefer(MyClass.col2)\n            )\n\n            # undefer all columns specific to a single class using Load + *\n            session.query(MyClass, MyOtherClass).options(\n                Load(MyClass).undefer("*"))\n\n            # undefer a column on a related object\n            session.query(MyClass).options(\n                defaultload(MyClass.items).undefer(MyClass.text))\n\n        :param key: Attribute to be undeferred.\n\n        .. seealso::\n\n            :ref:`orm_queryguide_column_deferral` - in the\n            :ref:`queryguide_toplevel`\n\n            :func:`_orm.defer`\n\n            :func:`_orm.undefer_group`\n\n        '
        return self._set_column_strategy((key,), {'deferred': False, 'instrument': True})

    def undefer_group(self, name: str) -> Self:
        if False:
            print('Hello World!')
        'Indicate that columns within the given deferred group name should be\n        undeferred.\n\n        The columns being undeferred are set up on the mapping as\n        :func:`.deferred` attributes and include a "group" name.\n\n        E.g::\n\n            session.query(MyClass).options(undefer_group("large_attrs"))\n\n        To undefer a group of attributes on a related entity, the path can be\n        spelled out using relationship loader options, such as\n        :func:`_orm.defaultload`::\n\n            session.query(MyClass).options(\n                defaultload("someattr").undefer_group("large_attrs"))\n\n        .. seealso::\n\n            :ref:`orm_queryguide_column_deferral` - in the\n            :ref:`queryguide_toplevel`\n\n            :func:`_orm.defer`\n\n            :func:`_orm.undefer`\n\n        '
        return self._set_column_strategy((_WILDCARD_TOKEN,), None, {f'undefer_group_{name}': True})

    def with_expression(self, key: _AttrType, expression: _ColumnExpressionArgument[Any]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Apply an ad-hoc SQL expression to a "deferred expression"\n        attribute.\n\n        This option is used in conjunction with the\n        :func:`_orm.query_expression` mapper-level construct that indicates an\n        attribute which should be the target of an ad-hoc SQL expression.\n\n        E.g.::\n\n            stmt = select(SomeClass).options(\n                with_expression(SomeClass.x_y_expr, SomeClass.x + SomeClass.y)\n            )\n\n        .. versionadded:: 1.2\n\n        :param key: Attribute to be populated\n\n        :param expr: SQL expression to be applied to the attribute.\n\n        .. seealso::\n\n            :ref:`orm_queryguide_with_expression` - background and usage\n            examples\n\n        '
        expression = _orm_full_deannotate(coercions.expect(roles.LabeledColumnExprRole, expression))
        return self._set_column_strategy((key,), {'query_expression': True}, extra_criteria=(expression,))

    def selectin_polymorphic(self, classes: Iterable[Type[Any]]) -> Self:
        if False:
            return 10
        'Indicate an eager load should take place for all attributes\n        specific to a subclass.\n\n        This uses an additional SELECT with IN against all matched primary\n        key values, and is the per-query analogue to the ``"selectin"``\n        setting on the :paramref:`.mapper.polymorphic_load` parameter.\n\n        .. versionadded:: 1.2\n\n        .. seealso::\n\n            :ref:`polymorphic_selectin`\n\n        '
        self = self._set_class_strategy({'selectinload_polymorphic': True}, opts={'entities': tuple(sorted((inspect(cls) for cls in classes), key=id))})
        return self

    @overload
    def _coerce_strat(self, strategy: _StrategySpec) -> _StrategyKey:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def _coerce_strat(self, strategy: Literal[None]) -> None:
        if False:
            print('Hello World!')
        ...

    def _coerce_strat(self, strategy: Optional[_StrategySpec]) -> Optional[_StrategyKey]:
        if False:
            return 10
        if strategy is not None:
            strategy_key = tuple(sorted(strategy.items()))
        else:
            strategy_key = None
        return strategy_key

    @_generative
    def _set_relationship_strategy(self, attr: _AttrType, strategy: Optional[_StrategySpec], propagate_to_loaders: bool=True, opts: Optional[_OptsType]=None, _reconcile_to_other: Optional[bool]=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy((attr,), strategy_key, _RELATIONSHIP_TOKEN, opts=opts, propagate_to_loaders=propagate_to_loaders, reconcile_to_other=_reconcile_to_other)
        return self

    @_generative
    def _set_column_strategy(self, attrs: Tuple[_AttrType, ...], strategy: Optional[_StrategySpec], opts: Optional[_OptsType]=None, extra_criteria: Optional[Tuple[Any, ...]]=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy(attrs, strategy_key, _COLUMN_TOKEN, opts=opts, attr_group=attrs, extra_criteria=extra_criteria)
        return self

    @_generative
    def _set_generic_strategy(self, attrs: Tuple[_AttrType, ...], strategy: _StrategySpec, _reconcile_to_other: Optional[bool]=None) -> Self:
        if False:
            while True:
                i = 10
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy(attrs, strategy_key, None, propagate_to_loaders=True, reconcile_to_other=_reconcile_to_other)
        return self

    @_generative
    def _set_class_strategy(self, strategy: _StrategySpec, opts: _OptsType) -> Self:
        if False:
            for i in range(10):
                print('nop')
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy(None, strategy_key, None, opts=opts)
        return self

    def _apply_to_parent(self, parent: Load) -> None:
        if False:
            return 10
        'apply this :class:`_orm._AbstractLoad` object as a sub-option o\n        a :class:`_orm.Load` object.\n\n        Implementation is provided by subclasses.\n\n        '
        raise NotImplementedError()

    def options(self, *opts: _AbstractLoad) -> Self:
        if False:
            while True:
                i = 10
        'Apply a series of options as sub-options to this\n        :class:`_orm._AbstractLoad` object.\n\n        Implementation is provided by subclasses.\n\n        '
        raise NotImplementedError()

    def _clone_for_bind_strategy(self, attrs: Optional[Tuple[_AttrType, ...]], strategy: Optional[_StrategyKey], wildcard_key: Optional[_WildcardKeyType], opts: Optional[_OptsType]=None, attr_group: Optional[_AttrGroupType]=None, propagate_to_loaders: bool=True, reconcile_to_other: Optional[bool]=None, extra_criteria: Optional[Tuple[Any, ...]]=None) -> Self:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def process_compile_state_replaced_entities(self, compile_state: ORMCompileState, mapper_entities: Sequence[_MapperEntity]) -> None:
        if False:
            return 10
        if not compile_state.compile_options._enable_eagerloads:
            return
        self._process(compile_state, mapper_entities, not bool(compile_state.current_path))

    def process_compile_state(self, compile_state: ORMCompileState) -> None:
        if False:
            print('Hello World!')
        if not compile_state.compile_options._enable_eagerloads:
            return
        self._process(compile_state, compile_state._lead_mapper_entities, not bool(compile_state.current_path) and (not compile_state.compile_options._for_refresh_state))

    def _process(self, compile_state: ORMCompileState, mapper_entities: Sequence[_MapperEntity], raiseerr: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'implemented by subclasses'
        raise NotImplementedError()

    @classmethod
    def _chop_path(cls, to_chop: _PathRepresentation, path: PathRegistry, debug: bool=False) -> Optional[_PathRepresentation]:
        if False:
            return 10
        i = -1
        for (i, (c_token, p_token)) in enumerate(zip(to_chop, path.natural_path)):
            if isinstance(c_token, str):
                if i == 0 and (c_token.endswith(f':{_DEFAULT_TOKEN}') or c_token.endswith(f':{_WILDCARD_TOKEN}')):
                    return to_chop
                elif c_token != f'{_RELATIONSHIP_TOKEN}:{_WILDCARD_TOKEN}' and c_token != p_token.key:
                    return None
            if c_token is p_token:
                continue
            elif isinstance(c_token, InspectionAttr) and insp_is_mapper(c_token) and insp_is_mapper(p_token) and c_token.isa(p_token):
                continue
            else:
                return None
        return to_chop[i + 1:]

class Load(_AbstractLoad):
    """Represents loader options which modify the state of a
    ORM-enabled :class:`_sql.Select` or a legacy :class:`_query.Query` in
    order to affect how various mapped attributes are loaded.

    The :class:`_orm.Load` object is in most cases used implicitly behind the
    scenes when one makes use of a query option like :func:`_orm.joinedload`,
    :func:`_orm.defer`, or similar.   It typically is not instantiated directly
    except for in some very specific cases.

    .. seealso::

        :ref:`orm_queryguide_relationship_per_entity_wildcard` - illustrates an
        example where direct use of :class:`_orm.Load` may be useful

    """
    __slots__ = ('path', 'context', 'additional_source_entities')
    _traverse_internals = [('path', visitors.ExtendedInternalTraversal.dp_has_cache_key), ('context', visitors.InternalTraversal.dp_has_cache_key_list), ('propagate_to_loaders', visitors.InternalTraversal.dp_boolean), ('additional_source_entities', visitors.InternalTraversal.dp_has_cache_key_list)]
    _cache_key_traversal = None
    path: PathRegistry
    context: Tuple[_LoadElement, ...]
    additional_source_entities: Tuple[_InternalEntityType[Any], ...]

    def __init__(self, entity: _EntityType[Any]):
        if False:
            for i in range(10):
                print('nop')
        insp = cast('Union[Mapper[Any], AliasedInsp[Any]]', inspect(entity))
        insp._post_inspect
        self.path = insp._path_registry
        self.context = ()
        self.propagate_to_loaders = False
        self.additional_source_entities = ()

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'Load({self.path[0]})'

    @classmethod
    def _construct_for_existing_path(cls, path: AbstractEntityRegistry) -> Load:
        if False:
            while True:
                i = 10
        load = cls.__new__(cls)
        load.path = path
        load.context = ()
        load.propagate_to_loaders = False
        load.additional_source_entities = ()
        return load

    def _adapt_cached_option_to_uncached_option(self, context: QueryContext, uncached_opt: ORMOption) -> ORMOption:
        if False:
            while True:
                i = 10
        return self._adjust_for_extra_criteria(context)

    def _prepend_path(self, path: PathRegistry) -> Load:
        if False:
            i = 10
            return i + 15
        cloned = self._clone()
        cloned.context = tuple((element._prepend_path(path) for element in self.context))
        return cloned

    def _adjust_for_extra_criteria(self, context: QueryContext) -> Load:
        if False:
            print('Hello World!')
        'Apply the current bound parameters in a QueryContext to all\n        occurrences "extra_criteria" stored within this ``Load`` object,\n        returning a new instance of this ``Load`` object.\n\n        '
        orig_query = context.compile_state.select_statement
        orig_cache_key: Optional[CacheKey] = None
        replacement_cache_key: Optional[CacheKey] = None
        found_crit = False

        def process(opt: _LoadElement) -> _LoadElement:
            if False:
                while True:
                    i = 10
            nonlocal orig_cache_key, replacement_cache_key, found_crit
            found_crit = True
            if orig_cache_key is None or replacement_cache_key is None:
                orig_cache_key = orig_query._generate_cache_key()
                replacement_cache_key = context.query._generate_cache_key()
                assert orig_cache_key is not None
                assert replacement_cache_key is not None
            opt._extra_criteria = tuple((replacement_cache_key._apply_params_to_element(orig_cache_key, crit) for crit in opt._extra_criteria))
            return opt
        new_context = tuple((process(value._clone()) if value._extra_criteria else value for value in self.context))
        if found_crit:
            cloned = self._clone()
            cloned.context = new_context
            return cloned
        else:
            return self

    def _reconcile_query_entities_with_us(self, mapper_entities, raiseerr):
        if False:
            for i in range(10):
                print('nop')
        'called at process time to allow adjustment of the root\n        entity inside of _LoadElement objects.\n\n        '
        path = self.path
        ezero = None
        for ent in mapper_entities:
            ezero = ent.entity_zero
            if ezero and orm_util._entity_corresponds_to(ezero, cast('_InternalEntityType[Any]', path[0])):
                return ezero
        return None

    def _process(self, compile_state: ORMCompileState, mapper_entities: Sequence[_MapperEntity], raiseerr: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        reconciled_lead_entity = self._reconcile_query_entities_with_us(mapper_entities, raiseerr)
        for loader in self.context:
            loader.process_compile_state(self, compile_state, mapper_entities, reconciled_lead_entity, raiseerr)

    def _apply_to_parent(self, parent: Load) -> None:
        if False:
            i = 10
            return i + 15
        'apply this :class:`_orm.Load` object as a sub-option of another\n        :class:`_orm.Load` object.\n\n        This method is used by the :meth:`_orm.Load.options` method.\n\n        '
        cloned = self._generate()
        assert cloned.propagate_to_loaders == self.propagate_to_loaders
        if not any((orm_util._entity_corresponds_to_use_path_impl(elem, cloned.path.odd_element(0)) for elem in (parent.path.odd_element(-1),) + parent.additional_source_entities)):
            if len(cloned.path) > 1:
                attrname = cloned.path[1]
                parent_entity = cloned.path[0]
            else:
                attrname = cloned.path[0]
                parent_entity = cloned.path[0]
            _raise_for_does_not_link(parent.path, attrname, parent_entity)
        cloned.path = PathRegistry.coerce(parent.path[0:-1] + cloned.path[:])
        if self.context:
            cloned.context = tuple((value._prepend_path_from(parent) for value in self.context))
        if cloned.context:
            parent.context += cloned.context
            parent.additional_source_entities += cloned.additional_source_entities

    @_generative
    def options(self, *opts: _AbstractLoad) -> Self:
        if False:
            print('Hello World!')
        'Apply a series of options as sub-options to this\n        :class:`_orm.Load`\n        object.\n\n        E.g.::\n\n            query = session.query(Author)\n            query = query.options(\n                        joinedload(Author.book).options(\n                            load_only(Book.summary, Book.excerpt),\n                            joinedload(Book.citations).options(\n                                joinedload(Citation.author)\n                            )\n                        )\n                    )\n\n        :param \\*opts: A series of loader option objects (ultimately\n         :class:`_orm.Load` objects) which should be applied to the path\n         specified by this :class:`_orm.Load` object.\n\n        .. versionadded:: 1.3.6\n\n        .. seealso::\n\n            :func:`.defaultload`\n\n            :ref:`orm_queryguide_relationship_sub_options`\n\n        '
        for opt in opts:
            try:
                opt._apply_to_parent(self)
            except AttributeError as ae:
                if not isinstance(opt, _AbstractLoad):
                    raise sa_exc.ArgumentError(f'Loader option {opt} is not compatible with the Load.options() method.') from ae
                else:
                    raise
        return self

    def _clone_for_bind_strategy(self, attrs: Optional[Tuple[_AttrType, ...]], strategy: Optional[_StrategyKey], wildcard_key: Optional[_WildcardKeyType], opts: Optional[_OptsType]=None, attr_group: Optional[_AttrGroupType]=None, propagate_to_loaders: bool=True, reconcile_to_other: Optional[bool]=None, extra_criteria: Optional[Tuple[Any, ...]]=None) -> Self:
        if False:
            i = 10
            return i + 15
        if propagate_to_loaders:
            self.propagate_to_loaders = True
        if self.path.is_token:
            raise sa_exc.ArgumentError('Wildcard token cannot be followed by another entity')
        elif path_is_property(self.path):
            if strategy:
                self.path.prop._strategy_lookup(self.path.prop, strategy[0])
            else:
                raise sa_exc.ArgumentError(f"Mapped attribute '{self.path.prop}' does not refer to a mapped entity")
        if attrs is None:
            load_element = _ClassStrategyLoad.create(self.path, None, strategy, wildcard_key, opts, propagate_to_loaders, attr_group=attr_group, reconcile_to_other=reconcile_to_other, extra_criteria=extra_criteria)
            if load_element:
                self.context += (load_element,)
                assert opts is not None
                self.additional_source_entities += cast('Tuple[_InternalEntityType[Any]]', opts['entities'])
        else:
            for attr in attrs:
                if isinstance(attr, str):
                    load_element = _TokenStrategyLoad.create(self.path, attr, strategy, wildcard_key, opts, propagate_to_loaders, attr_group=attr_group, reconcile_to_other=reconcile_to_other, extra_criteria=extra_criteria)
                else:
                    load_element = _AttributeStrategyLoad.create(self.path, attr, strategy, wildcard_key, opts, propagate_to_loaders, attr_group=attr_group, reconcile_to_other=reconcile_to_other, extra_criteria=extra_criteria)
                if load_element:
                    if wildcard_key is _RELATIONSHIP_TOKEN:
                        self.path = load_element.path
                    self.context += (load_element,)
                    if load_element.local_opts.get('recursion_depth', False):
                        r1 = load_element._recurse()
                        self.context += (r1,)
        return self

    def __getstate__(self):
        if False:
            print('Hello World!')
        d = self._shallow_to_dict()
        d['path'] = self.path.serialize()
        return d

    def __setstate__(self, state):
        if False:
            return 10
        state['path'] = PathRegistry.deserialize(state['path'])
        self._shallow_from_dict(state)

class _WildcardLoad(_AbstractLoad):
    """represent a standalone '*' load operation"""
    __slots__ = ('strategy', 'path', 'local_opts')
    _traverse_internals = [('strategy', visitors.ExtendedInternalTraversal.dp_plain_obj), ('path', visitors.ExtendedInternalTraversal.dp_plain_obj), ('local_opts', visitors.ExtendedInternalTraversal.dp_string_multi_dict)]
    cache_key_traversal: _CacheKeyTraversalType = None
    strategy: Optional[Tuple[Any, ...]]
    local_opts: _OptsType
    path: Union[Tuple[()], Tuple[str]]
    propagate_to_loaders = False

    def __init__(self) -> None:
        if False:
            return 10
        self.path = ()
        self.strategy = None
        self.local_opts = util.EMPTY_DICT

    def _clone_for_bind_strategy(self, attrs, strategy, wildcard_key, opts=None, attr_group=None, propagate_to_loaders=True, reconcile_to_other=None, extra_criteria=None):
        if False:
            return 10
        assert attrs is not None
        attr = attrs[0]
        assert wildcard_key and isinstance(attr, str) and (attr in (_WILDCARD_TOKEN, _DEFAULT_TOKEN))
        attr = f'{wildcard_key}:{attr}'
        self.strategy = strategy
        self.path = (attr,)
        if opts:
            self.local_opts = util.immutabledict(opts)
        assert extra_criteria is None

    def options(self, *opts: _AbstractLoad) -> Self:
        if False:
            while True:
                i = 10
        raise NotImplementedError('Star option does not support sub-options')

    def _apply_to_parent(self, parent: Load) -> None:
        if False:
            i = 10
            return i + 15
        "apply this :class:`_orm._WildcardLoad` object as a sub-option of\n        a :class:`_orm.Load` object.\n\n        This method is used by the :meth:`_orm.Load.options` method.   Note\n        that :class:`_orm.WildcardLoad` itself can't have sub-options, but\n        it may be used as the sub-option of a :class:`_orm.Load` object.\n\n        "
        assert self.path
        attr = self.path[0]
        if attr.endswith(_DEFAULT_TOKEN):
            attr = f"{attr.split(':')[0]}:{_WILDCARD_TOKEN}"
        effective_path = cast(AbstractEntityRegistry, parent.path).token(attr)
        assert effective_path.is_token
        loader = _TokenStrategyLoad.create(effective_path, None, self.strategy, None, self.local_opts, self.propagate_to_loaders)
        parent.context += (loader,)

    def _process(self, compile_state, mapper_entities, raiseerr):
        if False:
            while True:
                i = 10
        is_refresh = compile_state.compile_options._for_refresh_state
        if is_refresh and (not self.propagate_to_loaders):
            return
        entities = [ent.entity_zero for ent in mapper_entities]
        current_path = compile_state.current_path
        start_path: _PathRepresentation = self.path
        if current_path:
            new_path = self._chop_path(start_path, current_path)
            if new_path is None:
                return
            assert new_path == start_path
        assert start_path and len(start_path) == 1
        token = start_path[0]
        assert isinstance(token, str)
        entity = self._find_entity_basestring(entities, token, raiseerr)
        if not entity:
            return
        path_element = entity
        assert isinstance(token, str)
        loader = _TokenStrategyLoad.create(path_element._path_registry, token, self.strategy, None, self.local_opts, self.propagate_to_loaders, raiseerr=raiseerr)
        if not loader:
            return
        assert loader.path.is_token
        loader.process_compile_state(self, compile_state, mapper_entities, None, raiseerr)
        return loader

    def _find_entity_basestring(self, entities: Iterable[_InternalEntityType[Any]], token: str, raiseerr: bool) -> Optional[_InternalEntityType[Any]]:
        if False:
            print('Hello World!')
        if token.endswith(f':{_WILDCARD_TOKEN}'):
            if len(list(entities)) != 1:
                if raiseerr:
                    raise sa_exc.ArgumentError(f"""Can't apply wildcard ('*') or load_only() loader option to multiple entities {', '.join((str(ent) for ent in entities))}. Specify loader options for each entity individually, such as {', '.join((f"Load({ent}).some_option('*')" for ent in entities))}.""")
        elif token.endswith(_DEFAULT_TOKEN):
            raiseerr = False
        for ent in entities:
            return ent
        else:
            if raiseerr:
                raise sa_exc.ArgumentError(f'''Query has only expression-based entities - can't find property named "{token}".''')
            else:
                return None

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        d = self._shallow_to_dict()
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        self._shallow_from_dict(state)

class _LoadElement(cache_key.HasCacheKey, traversals.HasShallowCopy, visitors.Traversible):
    """represents strategy information to select for a LoaderStrategy
    and pass options to it.

    :class:`._LoadElement` objects provide the inner datastructure
    stored by a :class:`_orm.Load` object and are also the object passed
    to methods like :meth:`.LoaderStrategy.setup_query`.

    .. versionadded:: 2.0

    """
    __slots__ = ('path', 'strategy', 'propagate_to_loaders', 'local_opts', '_extra_criteria', '_reconcile_to_other')
    __visit_name__ = 'load_element'
    _traverse_internals = [('path', visitors.ExtendedInternalTraversal.dp_has_cache_key), ('strategy', visitors.ExtendedInternalTraversal.dp_plain_obj), ('local_opts', visitors.ExtendedInternalTraversal.dp_string_multi_dict), ('_extra_criteria', visitors.InternalTraversal.dp_clauseelement_list), ('propagate_to_loaders', visitors.InternalTraversal.dp_plain_obj), ('_reconcile_to_other', visitors.InternalTraversal.dp_plain_obj)]
    _cache_key_traversal = None
    _extra_criteria: Tuple[Any, ...]
    _reconcile_to_other: Optional[bool]
    strategy: Optional[_StrategyKey]
    path: PathRegistry
    propagate_to_loaders: bool
    local_opts: util.immutabledict[str, Any]
    is_token_strategy: bool
    is_class_strategy: bool

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return id(self)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return traversals.compare(self, other)

    @property
    def is_opts_only(self) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self.local_opts and self.strategy is None)

    def _clone(self, **kw: Any) -> _LoadElement:
        if False:
            while True:
                i = 10
        cls = self.__class__
        s = cls.__new__(cls)
        self._shallow_copy_to(s)
        return s

    def _update_opts(self, **kw: Any) -> _LoadElement:
        if False:
            return 10
        new = self._clone()
        new.local_opts = new.local_opts.union(kw)
        return new

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            return 10
        d = self._shallow_to_dict()
        d['path'] = self.path.serialize()
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        state['path'] = PathRegistry.deserialize(state['path'])
        self._shallow_from_dict(state)

    def _raise_for_no_match(self, parent_loader, mapper_entities):
        if False:
            for i in range(10):
                print('nop')
        path = parent_loader.path
        found_entities = False
        for ent in mapper_entities:
            ezero = ent.entity_zero
            if ezero:
                found_entities = True
                break
        if not found_entities:
            raise sa_exc.ArgumentError(f"Query has only expression-based entities; attribute loader options for {path[0]} can't be applied here.")
        else:
            raise sa_exc.ArgumentError(f"Mapped class {path[0]} does not apply to any of the root entities in this query, e.g. {', '.join((str(x.entity_zero) for x in mapper_entities if x.entity_zero))}. Please specify the full path from one of the root entities to the target attribute. ")

    def _adjust_effective_path_for_current_path(self, effective_path: PathRegistry, current_path: PathRegistry) -> Optional[PathRegistry]:
        if False:
            return 10
        "receives the 'current_path' entry from an :class:`.ORMCompileState`\n        instance, which is set during lazy loads and secondary loader strategy\n        loads, and adjusts the given path to be relative to the\n        current_path.\n\n        E.g. given a loader path and current path::\n\n            lp: User -> orders -> Order -> items -> Item -> keywords -> Keyword\n\n            cp: User -> orders -> Order -> items\n\n        The adjusted path would be::\n\n            Item -> keywords -> Keyword\n\n\n        "
        chopped_start_path = Load._chop_path(effective_path.natural_path, current_path)
        if not chopped_start_path:
            return None
        tokens_removed_from_start_path = len(effective_path) - len(chopped_start_path)
        loader_lead_path_element = self.path[tokens_removed_from_start_path]
        effective_path = PathRegistry.coerce((loader_lead_path_element,) + chopped_start_path[1:])
        return effective_path

    def _init_path(self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria):
        if False:
            return 10
        'Apply ORM attributes and/or wildcard to an existing path, producing\n        a new path.\n\n        This method is used within the :meth:`.create` method to initialize\n        a :class:`._LoadElement` object.\n\n        '
        raise NotImplementedError()

    def _prepare_for_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        if False:
            while True:
                i = 10
        'implemented by subclasses.'
        raise NotImplementedError()

    def process_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        if False:
            return 10
        'populate ORMCompileState.attributes with loader state for this\n        _LoadElement.\n\n        '
        keys = self._prepare_for_compile_state(parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr)
        for key in keys:
            if key in compile_state.attributes:
                compile_state.attributes[key] = _LoadElement._reconcile(self, compile_state.attributes[key])
            else:
                compile_state.attributes[key] = self

    @classmethod
    def create(cls, path: PathRegistry, attr: Union[_AttrType, _StrPathToken, None], strategy: Optional[_StrategyKey], wildcard_key: Optional[_WildcardKeyType], local_opts: Optional[_OptsType], propagate_to_loaders: bool, raiseerr: bool=True, attr_group: Optional[_AttrGroupType]=None, reconcile_to_other: Optional[bool]=None, extra_criteria: Optional[Tuple[Any, ...]]=None) -> _LoadElement:
        if False:
            while True:
                i = 10
        'Create a new :class:`._LoadElement` object.'
        opt = cls.__new__(cls)
        opt.path = path
        opt.strategy = strategy
        opt.propagate_to_loaders = propagate_to_loaders
        opt.local_opts = util.immutabledict(local_opts) if local_opts else util.EMPTY_DICT
        opt._extra_criteria = ()
        if reconcile_to_other is not None:
            opt._reconcile_to_other = reconcile_to_other
        elif strategy is None and (not local_opts):
            opt._reconcile_to_other = True
        else:
            opt._reconcile_to_other = None
        path = opt._init_path(path, attr, wildcard_key, attr_group, raiseerr, extra_criteria)
        if not path:
            return None
        assert opt.is_token_strategy == path.is_token
        opt.path = path
        return opt

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def _recurse(self) -> _LoadElement:
        if False:
            for i in range(10):
                print('nop')
        cloned = self._clone()
        cloned.path = PathRegistry.coerce(self.path[:] + self.path[-2:])
        return cloned

    def _prepend_path_from(self, parent: Load) -> _LoadElement:
        if False:
            return 10
        "adjust the path of this :class:`._LoadElement` to be\n        a subpath of that of the given parent :class:`_orm.Load` object's\n        path.\n\n        This is used by the :meth:`_orm.Load._apply_to_parent` method,\n        which is in turn part of the :meth:`_orm.Load.options` method.\n\n        "
        if not any((orm_util._entity_corresponds_to_use_path_impl(elem, self.path.odd_element(0)) for elem in (parent.path.odd_element(-1),) + parent.additional_source_entities)):
            raise sa_exc.ArgumentError(f'Attribute "{self.path[1]}" does not link from element "{parent.path[-1]}".')
        return self._prepend_path(parent.path)

    def _prepend_path(self, path: PathRegistry) -> _LoadElement:
        if False:
            while True:
                i = 10
        cloned = self._clone()
        assert cloned.strategy == self.strategy
        assert cloned.local_opts == self.local_opts
        assert cloned.is_class_strategy == self.is_class_strategy
        cloned.path = PathRegistry.coerce(path[0:-1] + cloned.path[:])
        return cloned

    @staticmethod
    def _reconcile(replacement: _LoadElement, existing: _LoadElement) -> _LoadElement:
        if False:
            return 10
        'define behavior for when two Load objects are to be put into\n        the context.attributes under the same key.\n\n        :param replacement: ``_LoadElement`` that seeks to replace the\n         existing one\n\n        :param existing: ``_LoadElement`` that is already present.\n\n        '
        if replacement._reconcile_to_other:
            return existing
        elif replacement._reconcile_to_other is False:
            return replacement
        elif existing._reconcile_to_other:
            return replacement
        elif existing._reconcile_to_other is False:
            return existing
        if existing is replacement:
            return replacement
        elif existing.strategy == replacement.strategy and existing.local_opts == replacement.local_opts:
            return replacement
        elif replacement.is_opts_only:
            existing = existing._clone()
            existing.local_opts = existing.local_opts.union(replacement.local_opts)
            existing._extra_criteria += replacement._extra_criteria
            return existing
        elif existing.is_opts_only:
            replacement = replacement._clone()
            replacement.local_opts = replacement.local_opts.union(existing.local_opts)
            replacement._extra_criteria += existing._extra_criteria
            return replacement
        elif replacement.path.is_token:
            return replacement
        raise sa_exc.InvalidRequestError(f'Loader strategies for {replacement.path} conflict')

class _AttributeStrategyLoad(_LoadElement):
    """Loader strategies against specific relationship or column paths.

    e.g.::

        joinedload(User.addresses)
        defer(Order.name)
        selectinload(User.orders).lazyload(Order.items)

    """
    __slots__ = ('_of_type', '_path_with_polymorphic_path')
    __visit_name__ = 'attribute_strategy_load_element'
    _traverse_internals = _LoadElement._traverse_internals + [('_of_type', visitors.ExtendedInternalTraversal.dp_multi), ('_path_with_polymorphic_path', visitors.ExtendedInternalTraversal.dp_has_cache_key)]
    _of_type: Union[Mapper[Any], AliasedInsp[Any], None]
    _path_with_polymorphic_path: Optional[PathRegistry]
    is_class_strategy = False
    is_token_strategy = False

    def _init_path(self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria):
        if False:
            while True:
                i = 10
        assert attr is not None
        self._of_type = None
        self._path_with_polymorphic_path = None
        (insp, _, prop) = _parse_attr_argument(attr)
        if insp.is_property:
            prop = attr
            path = path[prop]
            if path.has_entity:
                path = path.entity_path
            return path
        elif not insp.is_attribute:
            assert False
        if not orm_util._entity_corresponds_to_use_path_impl(path[-1], attr.parent):
            if raiseerr:
                if attr_group and attr is not attr_group[0]:
                    raise sa_exc.ArgumentError("Can't apply wildcard ('*') or load_only() loader option to multiple entities in the same option. Use separate options per entity.")
                else:
                    _raise_for_does_not_link(path, str(attr), attr.parent)
            else:
                return None
        if extra_criteria:
            assert not attr._extra_criteria
            self._extra_criteria = extra_criteria
        else:
            self._extra_criteria = attr._extra_criteria
        if getattr(attr, '_of_type', None):
            ac = attr._of_type
            ext_info = inspect(ac)
            self._of_type = ext_info
            self._path_with_polymorphic_path = path.entity_path[prop]
            path = path[prop][ext_info]
        else:
            path = path[prop]
        if path.has_entity:
            path = path.entity_path
        return path

    def _generate_extra_criteria(self, context):
        if False:
            while True:
                i = 10
        'Apply the current bound parameters in a QueryContext to the\n        immediate "extra_criteria" stored with this Load object.\n\n        Load objects are typically pulled from the cached version of\n        the statement from a QueryContext.  The statement currently being\n        executed will have new values (and keys) for bound parameters in the\n        extra criteria which need to be applied by loader strategies when\n        they handle this criteria for a result set.\n\n        '
        assert self._extra_criteria, 'this should only be called if _extra_criteria is present'
        orig_query = context.compile_state.select_statement
        current_query = context.query
        k1 = orig_query._generate_cache_key()
        k2 = current_query._generate_cache_key()
        return k2._apply_params_to_element(k1, and_(*self._extra_criteria))

    def _set_of_type_info(self, context, current_path):
        if False:
            return 10
        assert self._path_with_polymorphic_path
        pwpi = self._of_type
        assert pwpi
        if not pwpi.is_aliased_class:
            pwpi = inspect(orm_util.AliasedInsp._with_polymorphic_factory(pwpi.mapper.base_mapper, (pwpi.mapper,), aliased=True, _use_mapper_path=True))
        start_path = self._path_with_polymorphic_path
        if current_path:
            new_path = self._adjust_effective_path_for_current_path(start_path, current_path)
            if new_path is None:
                return
            start_path = new_path
        key = ('path_with_polymorphic', start_path.natural_path)
        if key in context:
            existing_aliased_insp = context[key]
            this_aliased_insp = pwpi
            new_aliased_insp = existing_aliased_insp._merge_with(this_aliased_insp)
            context[key] = new_aliased_insp
        else:
            context[key] = pwpi

    def _prepare_for_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        if False:
            while True:
                i = 10
        current_path = compile_state.current_path
        is_refresh = compile_state.compile_options._for_refresh_state
        assert not self.path.is_token
        if is_refresh and (not self.propagate_to_loaders):
            return []
        if self._of_type:
            self._set_of_type_info(compile_state.attributes, current_path)
        if not self.strategy and (not self.local_opts):
            return []
        if raiseerr and (not reconciled_lead_entity):
            self._raise_for_no_match(parent_loader, mapper_entities)
        if self.path.has_entity:
            effective_path = self.path.parent
        else:
            effective_path = self.path
        if current_path:
            assert effective_path is not None
            effective_path = self._adjust_effective_path_for_current_path(effective_path, current_path)
            if effective_path is None:
                return []
        return [('loader', cast(PathRegistry, effective_path).natural_path)]

    def __getstate__(self):
        if False:
            while True:
                i = 10
        d = super().__getstate__()
        d['_extra_criteria'] = ()
        if self._path_with_polymorphic_path:
            d['_path_with_polymorphic_path'] = self._path_with_polymorphic_path.serialize()
        if self._of_type:
            if self._of_type.is_aliased_class:
                d['_of_type'] = None
            elif self._of_type.is_mapper:
                d['_of_type'] = self._of_type.class_
            else:
                assert False, 'unexpected object for _of_type'
        return d

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        super().__setstate__(state)
        if state.get('_path_with_polymorphic_path', None):
            self._path_with_polymorphic_path = PathRegistry.deserialize(state['_path_with_polymorphic_path'])
        else:
            self._path_with_polymorphic_path = None
        if state.get('_of_type', None):
            self._of_type = inspect(state['_of_type'])
        else:
            self._of_type = None

class _TokenStrategyLoad(_LoadElement):
    """Loader strategies against wildcard attributes

    e.g.::

        raiseload('*')
        Load(User).lazyload('*')
        defer('*')
        load_only(User.name, User.email)  # will create a defer('*')
        joinedload(User.addresses).raiseload('*')

    """
    __visit_name__ = 'token_strategy_load_element'
    inherit_cache = True
    is_class_strategy = False
    is_token_strategy = True

    def _init_path(self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria):
        if False:
            print('Hello World!')
        if attr is not None:
            default_token = attr.endswith(_DEFAULT_TOKEN)
            if attr.endswith(_WILDCARD_TOKEN) or default_token:
                if wildcard_key:
                    attr = f'{wildcard_key}:{attr}'
                path = path.token(attr)
                return path
            else:
                raise sa_exc.ArgumentError('Strings are not accepted for attribute names in loader options; please use class-bound attributes directly.')
        return path

    def _prepare_for_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        if False:
            print('Hello World!')
        current_path = compile_state.current_path
        is_refresh = compile_state.compile_options._for_refresh_state
        assert self.path.is_token
        if is_refresh and (not self.propagate_to_loaders):
            return []
        if not self.strategy and (not self.local_opts):
            return []
        effective_path = self.path
        if reconciled_lead_entity:
            effective_path = PathRegistry.coerce((reconciled_lead_entity,) + effective_path.path[1:])
        if current_path:
            new_effective_path = self._adjust_effective_path_for_current_path(effective_path, current_path)
            if new_effective_path is None:
                return []
            effective_path = new_effective_path
        return [('loader', natural_path) for natural_path in cast(TokenRegistry, effective_path)._generate_natural_for_superclasses()]

class _ClassStrategyLoad(_LoadElement):
    """Loader strategies that deals with a class as a target, not
    an attribute path

    e.g.::

        q = s.query(Person).options(
            selectin_polymorphic(Person, [Engineer, Manager])
        )

    """
    inherit_cache = True
    is_class_strategy = True
    is_token_strategy = False
    __visit_name__ = 'class_strategy_load_element'

    def _init_path(self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria):
        if False:
            while True:
                i = 10
        return path

    def _prepare_for_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        if False:
            while True:
                i = 10
        current_path = compile_state.current_path
        is_refresh = compile_state.compile_options._for_refresh_state
        if is_refresh and (not self.propagate_to_loaders):
            return []
        if not self.strategy and (not self.local_opts):
            return []
        effective_path = self.path
        if current_path:
            new_effective_path = self._adjust_effective_path_for_current_path(effective_path, current_path)
            if new_effective_path is None:
                return []
            effective_path = new_effective_path
        return [('loader', effective_path.natural_path)]

def _generate_from_keys(meth: Callable[..., _AbstractLoad], keys: Tuple[_AttrType, ...], chained: bool, kw: Any) -> _AbstractLoad:
    if False:
        while True:
            i = 10
    lead_element: Optional[_AbstractLoad] = None
    attr: Any
    for (is_default, _keys) in ((True, keys[0:-1]), (False, keys[-1:])):
        for attr in _keys:
            if isinstance(attr, str):
                if attr.startswith('.' + _WILDCARD_TOKEN):
                    util.warn_deprecated('The undocumented `.{WILDCARD}` format is deprecated and will be removed in a future version as it is believed to be unused. If you have been using this functionality, please comment on Issue #4390 on the SQLAlchemy project tracker.', version='1.4')
                    attr = attr[1:]
                if attr == _WILDCARD_TOKEN:
                    if is_default:
                        raise sa_exc.ArgumentError('Wildcard token cannot be followed by another entity')
                    if lead_element is None:
                        lead_element = _WildcardLoad()
                    lead_element = meth(lead_element, _DEFAULT_TOKEN, **kw)
                else:
                    raise sa_exc.ArgumentError('Strings are not accepted for attribute names in loader options; please use class-bound attributes directly.')
            else:
                if lead_element is None:
                    (_, lead_entity, _) = _parse_attr_argument(attr)
                    lead_element = Load(lead_entity)
                if is_default:
                    if not chained:
                        lead_element = lead_element.defaultload(attr)
                    else:
                        lead_element = meth(lead_element, attr, _is_chain=True, **kw)
                else:
                    lead_element = meth(lead_element, attr, **kw)
    assert lead_element
    return lead_element

def _parse_attr_argument(attr: _AttrType) -> Tuple[InspectionAttr, _InternalEntityType[Any], MapperProperty[Any]]:
    if False:
        return 10
    'parse an attribute or wildcard argument to produce an\n    :class:`._AbstractLoad` instance.\n\n    This is used by the standalone loader strategy functions like\n    ``joinedload()``, ``defer()``, etc. to produce :class:`_orm.Load` or\n    :class:`._WildcardLoad` objects.\n\n    '
    try:
        insp: InspectionAttr = inspect(attr)
    except sa_exc.NoInspectionAvailable as err:
        raise sa_exc.ArgumentError('expected ORM mapped attribute for loader strategy argument') from err
    lead_entity: _InternalEntityType[Any]
    if insp_is_mapper_property(insp):
        lead_entity = insp.parent
        prop = insp
    elif insp_is_attribute(insp):
        lead_entity = insp.parent
        prop = insp.prop
    else:
        raise sa_exc.ArgumentError('expected ORM mapped attribute for loader strategy argument')
    return (insp, lead_entity, prop)

def loader_unbound_fn(fn: _FN) -> _FN:
    if False:
        return 10
    'decorator that applies docstrings between standalone loader functions\n    and the loader methods on :class:`._AbstractLoad`.\n\n    '
    bound_fn = getattr(_AbstractLoad, fn.__name__)
    fn_doc = bound_fn.__doc__
    bound_fn.__doc__ = f'Produce a new :class:`_orm.Load` object with the\n:func:`_orm.{fn.__name__}` option applied.\n\nSee :func:`_orm.{fn.__name__}` for usage examples.\n\n'
    fn.__doc__ = fn_doc
    return fn

@loader_unbound_fn
def contains_eager(*keys: _AttrType, **kw: Any) -> _AbstractLoad:
    if False:
        i = 10
        return i + 15
    return _generate_from_keys(Load.contains_eager, keys, True, kw)

@loader_unbound_fn
def load_only(*attrs: _AttrType, raiseload: bool=False) -> _AbstractLoad:
    if False:
        i = 10
        return i + 15
    (_, lead_element, _) = _parse_attr_argument(attrs[0])
    return Load(lead_element).load_only(*attrs, raiseload=raiseload)

@loader_unbound_fn
def joinedload(*keys: _AttrType, **kw: Any) -> _AbstractLoad:
    if False:
        print('Hello World!')
    return _generate_from_keys(Load.joinedload, keys, False, kw)

@loader_unbound_fn
def subqueryload(*keys: _AttrType) -> _AbstractLoad:
    if False:
        return 10
    return _generate_from_keys(Load.subqueryload, keys, False, {})

@loader_unbound_fn
def selectinload(*keys: _AttrType, recursion_depth: Optional[int]=None) -> _AbstractLoad:
    if False:
        return 10
    return _generate_from_keys(Load.selectinload, keys, False, {'recursion_depth': recursion_depth})

@loader_unbound_fn
def lazyload(*keys: _AttrType) -> _AbstractLoad:
    if False:
        while True:
            i = 10
    return _generate_from_keys(Load.lazyload, keys, False, {})

@loader_unbound_fn
def immediateload(*keys: _AttrType, recursion_depth: Optional[int]=None) -> _AbstractLoad:
    if False:
        return 10
    return _generate_from_keys(Load.immediateload, keys, False, {'recursion_depth': recursion_depth})

@loader_unbound_fn
def noload(*keys: _AttrType) -> _AbstractLoad:
    if False:
        for i in range(10):
            print('nop')
    return _generate_from_keys(Load.noload, keys, False, {})

@loader_unbound_fn
def raiseload(*keys: _AttrType, **kw: Any) -> _AbstractLoad:
    if False:
        return 10
    return _generate_from_keys(Load.raiseload, keys, False, kw)

@loader_unbound_fn
def defaultload(*keys: _AttrType) -> _AbstractLoad:
    if False:
        return 10
    return _generate_from_keys(Load.defaultload, keys, False, {})

@loader_unbound_fn
def defer(key: _AttrType, *addl_attrs: _AttrType, raiseload: bool=False) -> _AbstractLoad:
    if False:
        i = 10
        return i + 15
    if addl_attrs:
        util.warn_deprecated('The *addl_attrs on orm.defer is deprecated.  Please use method chaining in conjunction with defaultload() to indicate a path.', version='1.3')
    if raiseload:
        kw = {'raiseload': raiseload}
    else:
        kw = {}
    return _generate_from_keys(Load.defer, (key,) + addl_attrs, False, kw)

@loader_unbound_fn
def undefer(key: _AttrType, *addl_attrs: _AttrType) -> _AbstractLoad:
    if False:
        print('Hello World!')
    if addl_attrs:
        util.warn_deprecated('The *addl_attrs on orm.undefer is deprecated.  Please use method chaining in conjunction with defaultload() to indicate a path.', version='1.3')
    return _generate_from_keys(Load.undefer, (key,) + addl_attrs, False, {})

@loader_unbound_fn
def undefer_group(name: str) -> _AbstractLoad:
    if False:
        for i in range(10):
            print('nop')
    element = _WildcardLoad()
    return element.undefer_group(name)

@loader_unbound_fn
def with_expression(key: _AttrType, expression: _ColumnExpressionArgument[Any]) -> _AbstractLoad:
    if False:
        for i in range(10):
            print('nop')
    return _generate_from_keys(Load.with_expression, (key,), False, {'expression': expression})

@loader_unbound_fn
def selectin_polymorphic(base_cls: _EntityType[Any], classes: Iterable[Type[Any]]) -> _AbstractLoad:
    if False:
        for i in range(10):
            print('nop')
    ul = Load(base_cls)
    return ul.selectin_polymorphic(classes)

def _raise_for_does_not_link(path, attrname, parent_entity):
    if False:
        print('Hello World!')
    if len(path) > 1:
        path_is_of_type = path[-1].entity is not path[-2].mapper.class_
        if insp_is_aliased_class(parent_entity):
            parent_entity_str = str(parent_entity)
        else:
            parent_entity_str = parent_entity.class_.__name__
        raise sa_exc.ArgumentError(f'ORM mapped entity or attribute "{attrname}" does not link from relationship "{path[-2]}%s".%s' % (f'.of_type({path[-1]})' if path_is_of_type else '', f'  Did you mean to use "{path[-2]}.of_type({parent_entity_str})" or "loadopt.options(selectin_polymorphic({path[-2].mapper.class_.__name__}, [{parent_entity_str}]), ...)" ?' if not path_is_of_type and (not path[-1].is_aliased_class) and orm_util._entity_corresponds_to(path.entity, inspect(parent_entity).mapper) else ''))
    else:
        raise sa_exc.ArgumentError(f'ORM mapped attribute "{attrname}" does not link mapped class "{path[-1]}"')