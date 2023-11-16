"""The internals for the unit of work system.

The session's flush() process passes objects to a contextual object
here, which assembles flush tasks based on mappers and their properties,
organizes them in order of dependency, and executes.

"""
from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological
if TYPE_CHECKING:
    from .dependency import DependencyProcessor
    from .interfaces import MapperProperty
    from .mapper import Mapper
    from .session import Session
    from .session import SessionTransaction
    from .state import InstanceState

def track_cascade_events(descriptor, prop):
    if False:
        return 10
    'Establish event listeners on object attributes which handle\n    cascade-on-set/append.\n\n    '
    key = prop.key

    def append(state, item, initiator, **kw):
        if False:
            while True:
                i = 10
        if item is None:
            return
        sess = state.session
        if sess:
            if sess._warn_on_events:
                sess._flush_warning('collection append')
            prop = state.manager.mapper._props[key]
            item_state = attributes.instance_state(item)
            if prop._cascade.save_update and key == initiator.key and (not sess._contains_state(item_state)):
                sess._save_or_update_state(item_state)
        return item

    def remove(state, item, initiator, **kw):
        if False:
            return 10
        if item is None:
            return
        sess = state.session
        prop = state.manager.mapper._props[key]
        if sess and sess._warn_on_events:
            sess._flush_warning('collection remove' if prop.uselist else 'related attribute delete')
        if item is not None and item is not attributes.NEVER_SET and (item is not attributes.PASSIVE_NO_RESULT) and prop._cascade.delete_orphan:
            item_state = attributes.instance_state(item)
            if prop.mapper._is_orphan(item_state):
                if sess and item_state in sess._new:
                    sess.expunge(item)
                else:
                    item_state._orphaned_outside_of_session = True

    def set_(state, newvalue, oldvalue, initiator, **kw):
        if False:
            while True:
                i = 10
        if oldvalue is newvalue:
            return newvalue
        sess = state.session
        if sess:
            if sess._warn_on_events:
                sess._flush_warning('related attribute set')
            prop = state.manager.mapper._props[key]
            if newvalue is not None:
                newvalue_state = attributes.instance_state(newvalue)
                if prop._cascade.save_update and key == initiator.key and (not sess._contains_state(newvalue_state)):
                    sess._save_or_update_state(newvalue_state)
            if oldvalue is not None and oldvalue is not attributes.NEVER_SET and (oldvalue is not attributes.PASSIVE_NO_RESULT) and prop._cascade.delete_orphan:
                oldvalue_state = attributes.instance_state(oldvalue)
                if oldvalue_state in sess._new and prop.mapper._is_orphan(oldvalue_state):
                    sess.expunge(oldvalue)
        return newvalue
    event.listen(descriptor, 'append_wo_mutation', append, raw=True, include_key=True)
    event.listen(descriptor, 'append', append, raw=True, retval=True, include_key=True)
    event.listen(descriptor, 'remove', remove, raw=True, retval=True, include_key=True)
    event.listen(descriptor, 'set', set_, raw=True, retval=True, include_key=True)

class UOWTransaction:
    session: Session
    transaction: SessionTransaction
    attributes: Dict[str, Any]
    deps: util.defaultdict[Mapper[Any], Set[DependencyProcessor]]
    mappers: util.defaultdict[Mapper[Any], Set[InstanceState[Any]]]

    def __init__(self, session: Session):
        if False:
            print('Hello World!')
        self.session = session
        self.attributes = {}
        self.deps = util.defaultdict(set)
        self.mappers = util.defaultdict(set)
        self.presort_actions = {}
        self.postsort_actions = {}
        self.dependencies = set()
        self.states = {}
        self.post_update_states = util.defaultdict(lambda : (set(), set()))

    @property
    def has_work(self):
        if False:
            return 10
        return bool(self.states)

    def was_already_deleted(self, state):
        if False:
            i = 10
            return i + 15
        'Return ``True`` if the given state is expired and was deleted\n        previously.\n        '
        if state.expired:
            try:
                state._load_expired(state, attributes.PASSIVE_OFF)
            except orm_exc.ObjectDeletedError:
                self.session._remove_newly_deleted([state])
                return True
        return False

    def is_deleted(self, state):
        if False:
            for i in range(10):
                print('nop')
        'Return ``True`` if the given state is marked as deleted\n        within this uowtransaction.'
        return state in self.states and self.states[state][0]

    def memo(self, key, callable_):
        if False:
            return 10
        if key in self.attributes:
            return self.attributes[key]
        else:
            self.attributes[key] = ret = callable_()
            return ret

    def remove_state_actions(self, state):
        if False:
            print('Hello World!')
        'Remove pending actions for a state from the uowtransaction.'
        isdelete = self.states[state][0]
        self.states[state] = (isdelete, True)

    def get_attribute_history(self, state, key, passive=attributes.PASSIVE_NO_INITIALIZE):
        if False:
            while True:
                i = 10
        'Facade to attributes.get_state_history(), including\n        caching of results.'
        hashkey = ('history', state, key)
        if hashkey in self.attributes:
            (history, state_history, cached_passive) = self.attributes[hashkey]
            if not cached_passive & attributes.SQL_OK and passive & attributes.SQL_OK:
                impl = state.manager[key].impl
                history = impl.get_history(state, state.dict, attributes.PASSIVE_OFF | attributes.LOAD_AGAINST_COMMITTED | attributes.NO_RAISE)
                if history and impl.uses_objects:
                    state_history = history.as_state()
                else:
                    state_history = history
                self.attributes[hashkey] = (history, state_history, passive)
        else:
            impl = state.manager[key].impl
            history = impl.get_history(state, state.dict, passive | attributes.LOAD_AGAINST_COMMITTED | attributes.NO_RAISE)
            if history and impl.uses_objects:
                state_history = history.as_state()
            else:
                state_history = history
            self.attributes[hashkey] = (history, state_history, passive)
        return state_history

    def has_dep(self, processor):
        if False:
            return 10
        return (processor, True) in self.presort_actions

    def register_preprocessor(self, processor, fromparent):
        if False:
            while True:
                i = 10
        key = (processor, fromparent)
        if key not in self.presort_actions:
            self.presort_actions[key] = Preprocess(processor, fromparent)

    def register_object(self, state: InstanceState[Any], isdelete: bool=False, listonly: bool=False, cancel_delete: bool=False, operation: Optional[str]=None, prop: Optional[MapperProperty]=None) -> bool:
        if False:
            while True:
                i = 10
        if not self.session._contains_state(state):
            if not state.deleted and operation is not None:
                util.warn("Object of type %s not in session, %s operation along '%s' will not proceed" % (orm_util.state_class_str(state), operation, prop))
            return False
        if state not in self.states:
            mapper = state.manager.mapper
            if mapper not in self.mappers:
                self._per_mapper_flush_actions(mapper)
            self.mappers[mapper].add(state)
            self.states[state] = (isdelete, listonly)
        elif not listonly and (isdelete or cancel_delete):
            self.states[state] = (isdelete, False)
        return True

    def register_post_update(self, state, post_update_cols):
        if False:
            print('Hello World!')
        mapper = state.manager.mapper.base_mapper
        (states, cols) = self.post_update_states[mapper]
        states.add(state)
        cols.update(post_update_cols)

    def _per_mapper_flush_actions(self, mapper):
        if False:
            print('Hello World!')
        saves = SaveUpdateAll(self, mapper.base_mapper)
        deletes = DeleteAll(self, mapper.base_mapper)
        self.dependencies.add((saves, deletes))
        for dep in mapper._dependency_processors:
            dep.per_property_preprocessors(self)
        for prop in mapper.relationships:
            if prop.viewonly:
                continue
            dep = prop._dependency_processor
            dep.per_property_preprocessors(self)

    @util.memoized_property
    def _mapper_for_dep(self):
        if False:
            i = 10
            return i + 15
        'return a dynamic mapping of (Mapper, DependencyProcessor) to\n        True or False, indicating if the DependencyProcessor operates\n        on objects of that Mapper.\n\n        The result is stored in the dictionary persistently once\n        calculated.\n\n        '
        return util.PopulateDict(lambda tup: tup[0]._props.get(tup[1].key) is tup[1].prop)

    def filter_states_for_dep(self, dep, states):
        if False:
            for i in range(10):
                print('nop')
        'Filter the given list of InstanceStates to those relevant to the\n        given DependencyProcessor.\n\n        '
        mapper_for_dep = self._mapper_for_dep
        return [s for s in states if mapper_for_dep[s.manager.mapper, dep]]

    def states_for_mapper_hierarchy(self, mapper, isdelete, listonly):
        if False:
            return 10
        checktup = (isdelete, listonly)
        for mapper in mapper.base_mapper.self_and_descendants:
            for state in self.mappers[mapper]:
                if self.states[state] == checktup:
                    yield state

    def _generate_actions(self):
        if False:
            print('Hello World!')
        'Generate the full, unsorted collection of PostSortRecs as\n        well as dependency pairs for this UOWTransaction.\n\n        '
        while True:
            ret = False
            for action in list(self.presort_actions.values()):
                if action.execute(self):
                    ret = True
            if not ret:
                break
        self.cycles = cycles = topological.find_cycles(self.dependencies, list(self.postsort_actions.values()))
        if cycles:
            convert = {rec: set(rec.per_state_flush_actions(self)) for rec in cycles}
            for edge in list(self.dependencies):
                if None in edge or edge[0].disabled or edge[1].disabled or cycles.issuperset(edge):
                    self.dependencies.remove(edge)
                elif edge[0] in cycles:
                    self.dependencies.remove(edge)
                    for dep in convert[edge[0]]:
                        self.dependencies.add((dep, edge[1]))
                elif edge[1] in cycles:
                    self.dependencies.remove(edge)
                    for dep in convert[edge[1]]:
                        self.dependencies.add((edge[0], dep))
        return {a for a in self.postsort_actions.values() if not a.disabled}.difference(cycles)

    def execute(self) -> None:
        if False:
            while True:
                i = 10
        postsort_actions = self._generate_actions()
        postsort_actions = sorted(postsort_actions, key=lambda item: item.sort_key)
        if self.cycles:
            for subset in topological.sort_as_subsets(self.dependencies, postsort_actions):
                set_ = set(subset)
                while set_:
                    n = set_.pop()
                    n.execute_aggregate(self, set_)
        else:
            for rec in topological.sort(self.dependencies, postsort_actions):
                rec.execute(self)

    def finalize_flush_changes(self) -> None:
        if False:
            i = 10
            return i + 15
        'Mark processed objects as clean / deleted after a successful\n        flush().\n\n        This method is called within the flush() method after the\n        execute() method has succeeded and the transaction has been committed.\n\n        '
        if not self.states:
            return
        states = set(self.states)
        isdel = {s for (s, (isdelete, listonly)) in self.states.items() if isdelete}
        other = states.difference(isdel)
        if isdel:
            self.session._remove_newly_deleted(isdel)
        if other:
            self.session._register_persistent(other)

class IterateMappersMixin:
    __slots__ = ()

    def _mappers(self, uow):
        if False:
            while True:
                i = 10
        if self.fromparent:
            return iter((m for m in self.dependency_processor.parent.self_and_descendants if uow._mapper_for_dep[m, self.dependency_processor]))
        else:
            return self.dependency_processor.mapper.self_and_descendants

class Preprocess(IterateMappersMixin):
    __slots__ = ('dependency_processor', 'fromparent', 'processed', 'setup_flush_actions')

    def __init__(self, dependency_processor, fromparent):
        if False:
            while True:
                i = 10
        self.dependency_processor = dependency_processor
        self.fromparent = fromparent
        self.processed = set()
        self.setup_flush_actions = False

    def execute(self, uow):
        if False:
            return 10
        delete_states = set()
        save_states = set()
        for mapper in self._mappers(uow):
            for state in uow.mappers[mapper].difference(self.processed):
                (isdelete, listonly) = uow.states[state]
                if not listonly:
                    if isdelete:
                        delete_states.add(state)
                    else:
                        save_states.add(state)
        if delete_states:
            self.dependency_processor.presort_deletes(uow, delete_states)
            self.processed.update(delete_states)
        if save_states:
            self.dependency_processor.presort_saves(uow, save_states)
            self.processed.update(save_states)
        if delete_states or save_states:
            if not self.setup_flush_actions and (self.dependency_processor.prop_has_changes(uow, delete_states, True) or self.dependency_processor.prop_has_changes(uow, save_states, False)):
                self.dependency_processor.per_property_flush_actions(uow)
                self.setup_flush_actions = True
            return True
        else:
            return False

class PostSortRec:
    __slots__ = ('disabled',)

    def __new__(cls, uow, *args):
        if False:
            for i in range(10):
                print('nop')
        key = (cls,) + args
        if key in uow.postsort_actions:
            return uow.postsort_actions[key]
        else:
            uow.postsort_actions[key] = ret = object.__new__(cls)
            ret.disabled = False
            return ret

    def execute_aggregate(self, uow, recs):
        if False:
            return 10
        self.execute(uow)

class ProcessAll(IterateMappersMixin, PostSortRec):
    __slots__ = ('dependency_processor', 'isdelete', 'fromparent', 'sort_key')

    def __init__(self, uow, dependency_processor, isdelete, fromparent):
        if False:
            for i in range(10):
                print('nop')
        self.dependency_processor = dependency_processor
        self.sort_key = ('ProcessAll', self.dependency_processor.sort_key, isdelete)
        self.isdelete = isdelete
        self.fromparent = fromparent
        uow.deps[dependency_processor.parent.base_mapper].add(dependency_processor)

    def execute(self, uow):
        if False:
            print('Hello World!')
        states = self._elements(uow)
        if self.isdelete:
            self.dependency_processor.process_deletes(uow, states)
        else:
            self.dependency_processor.process_saves(uow, states)

    def per_state_flush_actions(self, uow):
        if False:
            i = 10
            return i + 15
        return iter([])

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%s, isdelete=%s)' % (self.__class__.__name__, self.dependency_processor, self.isdelete)

    def _elements(self, uow):
        if False:
            return 10
        for mapper in self._mappers(uow):
            for state in uow.mappers[mapper]:
                (isdelete, listonly) = uow.states[state]
                if isdelete == self.isdelete and (not listonly):
                    yield state

class PostUpdateAll(PostSortRec):
    __slots__ = ('mapper', 'isdelete', 'sort_key')

    def __init__(self, uow, mapper, isdelete):
        if False:
            print('Hello World!')
        self.mapper = mapper
        self.isdelete = isdelete
        self.sort_key = ('PostUpdateAll', mapper._sort_key, isdelete)

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute(self, uow):
        if False:
            i = 10
            return i + 15
        persistence = util.preloaded.orm_persistence
        (states, cols) = uow.post_update_states[self.mapper]
        states = [s for s in states if uow.states[s][0] == self.isdelete]
        persistence.post_update(self.mapper, states, uow, cols)

class SaveUpdateAll(PostSortRec):
    __slots__ = ('mapper', 'sort_key')

    def __init__(self, uow, mapper):
        if False:
            return 10
        self.mapper = mapper
        self.sort_key = ('SaveUpdateAll', mapper._sort_key)
        assert mapper is mapper.base_mapper

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute(self, uow):
        if False:
            print('Hello World!')
        util.preloaded.orm_persistence.save_obj(self.mapper, uow.states_for_mapper_hierarchy(self.mapper, False, False), uow)

    def per_state_flush_actions(self, uow):
        if False:
            i = 10
            return i + 15
        states = list(uow.states_for_mapper_hierarchy(self.mapper, False, False))
        base_mapper = self.mapper.base_mapper
        delete_all = DeleteAll(uow, base_mapper)
        for state in states:
            action = SaveUpdateState(uow, state)
            uow.dependencies.add((action, delete_all))
            yield action
        for dep in uow.deps[self.mapper]:
            states_for_prop = uow.filter_states_for_dep(dep, states)
            dep.per_state_flush_actions(uow, states_for_prop, False)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%s)' % (self.__class__.__name__, self.mapper)

class DeleteAll(PostSortRec):
    __slots__ = ('mapper', 'sort_key')

    def __init__(self, uow, mapper):
        if False:
            print('Hello World!')
        self.mapper = mapper
        self.sort_key = ('DeleteAll', mapper._sort_key)
        assert mapper is mapper.base_mapper

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute(self, uow):
        if False:
            return 10
        util.preloaded.orm_persistence.delete_obj(self.mapper, uow.states_for_mapper_hierarchy(self.mapper, True, False), uow)

    def per_state_flush_actions(self, uow):
        if False:
            print('Hello World!')
        states = list(uow.states_for_mapper_hierarchy(self.mapper, True, False))
        base_mapper = self.mapper.base_mapper
        save_all = SaveUpdateAll(uow, base_mapper)
        for state in states:
            action = DeleteState(uow, state)
            uow.dependencies.add((save_all, action))
            yield action
        for dep in uow.deps[self.mapper]:
            states_for_prop = uow.filter_states_for_dep(dep, states)
            dep.per_state_flush_actions(uow, states_for_prop, True)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '%s(%s)' % (self.__class__.__name__, self.mapper)

class ProcessState(PostSortRec):
    __slots__ = ('dependency_processor', 'isdelete', 'state', 'sort_key')

    def __init__(self, uow, dependency_processor, isdelete, state):
        if False:
            print('Hello World!')
        self.dependency_processor = dependency_processor
        self.sort_key = ('ProcessState', dependency_processor.sort_key)
        self.isdelete = isdelete
        self.state = state

    def execute_aggregate(self, uow, recs):
        if False:
            while True:
                i = 10
        cls_ = self.__class__
        dependency_processor = self.dependency_processor
        isdelete = self.isdelete
        our_recs = [r for r in recs if r.__class__ is cls_ and r.dependency_processor is dependency_processor and (r.isdelete is isdelete)]
        recs.difference_update(our_recs)
        states = [self.state] + [r.state for r in our_recs]
        if isdelete:
            dependency_processor.process_deletes(uow, states)
        else:
            dependency_processor.process_saves(uow, states)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s(%s, %s, delete=%s)' % (self.__class__.__name__, self.dependency_processor, orm_util.state_str(self.state), self.isdelete)

class SaveUpdateState(PostSortRec):
    __slots__ = ('state', 'mapper', 'sort_key')

    def __init__(self, uow, state):
        if False:
            i = 10
            return i + 15
        self.state = state
        self.mapper = state.mapper.base_mapper
        self.sort_key = ('ProcessState', self.mapper._sort_key)

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute_aggregate(self, uow, recs):
        if False:
            print('Hello World!')
        persistence = util.preloaded.orm_persistence
        cls_ = self.__class__
        mapper = self.mapper
        our_recs = [r for r in recs if r.__class__ is cls_ and r.mapper is mapper]
        recs.difference_update(our_recs)
        persistence.save_obj(mapper, [self.state] + [r.state for r in our_recs], uow)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%s)' % (self.__class__.__name__, orm_util.state_str(self.state))

class DeleteState(PostSortRec):
    __slots__ = ('state', 'mapper', 'sort_key')

    def __init__(self, uow, state):
        if False:
            print('Hello World!')
        self.state = state
        self.mapper = state.mapper.base_mapper
        self.sort_key = ('DeleteState', self.mapper._sort_key)

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute_aggregate(self, uow, recs):
        if False:
            for i in range(10):
                print('nop')
        persistence = util.preloaded.orm_persistence
        cls_ = self.__class__
        mapper = self.mapper
        our_recs = [r for r in recs if r.__class__ is cls_ and r.mapper is mapper]
        recs.difference_update(our_recs)
        states = [self.state] + [r.state for r in our_recs]
        persistence.delete_obj(mapper, [s for s in states if uow.states[s][0]], uow)

    def __repr__(self):
        if False:
            return 10
        return '%s(%s)' % (self.__class__.__name__, orm_util.state_str(self.state))