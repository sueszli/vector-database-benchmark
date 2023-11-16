from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
from typing import Optional
import marshal
import random
import weakref
import re
import sys
import time
import io
import types
import copyreg
import functools
import renpy

class StoreDeleted(object):

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        if PY2:
            return b'deleted'
        else:
            return 'deleted'
deleted = StoreDeleted()

class SlottedNoRollback(object):
    """
    :doc: norollback class

    Instances of classes inheriting from this class do not participate
    in rollback. The difference between this and :class:`NoRollback` is that
    this class does not have an associated dictionary, hence can be used
    with ``__slots__`` to reduce memory usage.

    Objects reachable through an instance of a NoRollback class only participate
    in rollback if they are reachable through other paths.
    """
    __slots__ = ()

class NoRollback(SlottedNoRollback):
    """
    :doc: norollback class

    Instances of this class, and classes inheriting from this class,
    do not participate in rollback. Objects reachable through an instance
    of a NoRollback class only participate in rollback if they are
    reachable through other paths.
    """

class AlwaysRollback(renpy.revertable.RevertableObject):
    """
    This is a revertable object that always participates in rollback.
    It's used when a revertable object is created by an object that
    doesn't participate in the rollback system.
    """

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self = super(AlwaysRollback, cls).__new__(cls)
        log = renpy.game.log
        if log is not None:
            del log.mutated[id(self)]
        return self
NOROLLBACK_TYPES = tuple()

def reached(obj, reachable, wait):
    if False:
        return 10
    '\n    @param obj: The object that was reached.\n\n    `reachable`\n        A map from id(obj) to int. The int is 1 if the object was reached\n        normally, and 0 if it was reached, but inherits from NoRollback.\n    '
    if wait:
        wait()
    idobj = id(obj)
    if idobj in reachable:
        return
    reachable[idobj] = obj
    if isinstance(obj, NOROLLBACK_TYPES):
        return
    try:
        nosave = getattr(obj, 'nosave', None)
        if nosave is not None:
            nosave = getattr(obj, 'noreach', nosave)
            for (k, v) in vars(obj).items():
                if k not in nosave:
                    reached(v, reachable, wait)
        else:
            for v in vars(obj).values():
                reached(v, reachable, wait)
    except Exception:
        pass
    try:
        if not len(obj) or isinstance(obj, basestring):
            return
    except Exception:
        return
    try:
        for v in obj.__iter__():
            reached(v, reachable, wait)
    except Exception:
        pass
    try:
        for v in obj.values():
            reached(v, reachable, wait)
    except Exception:
        pass

def reached_vars(store, reachable, wait):
    if False:
        print('Hello World!')
    '\n    Marks everything reachable from the variables in the store\n    or from the context info objects as reachable.\n\n    `store`\n        A map from variable name to variable value.\n\n    `reachable`\n        A dictionary that will be filled in with a map from id(obj) to obj.\n    '
    for v in store.values():
        reached(v, reachable, wait)
    for c in renpy.game.contexts:
        reached(c.info, reachable, wait)
        reached(c.music, reachable, wait)
        for d in c.dynamic_stack:
            for v in d.values():
                reached(v, reachable, wait)
generation = time.time()
serial = 0
rng = renpy.revertable.DetRandom()

class Rollback(renpy.object.Object):
    """
    Allows the state of the game to be rolled back to the point just
    before a node began executing.

    @ivar context: A shallow copy of the context we were in before
    we started executing the node. (Shallow copy also includes
    a copy of the associated SceneList.)

    @ivar objects: A list of tuples, each containing an object and a
    token of information that, when passed to the rollback method on
    that object, causes that object to rollback.

    @ivar store: A list of updates to store that will cause the state
    of the store to be rolled back to the start of node
    execution. This is a list of tuples, either (key, value) tuples
    representing a value that needs to be assigned to a key, or (key,)
    tuples that mean the key should be deleted.

    @ivar checkpoint: True if this is a user-visible checkpoint,
    false otherwise.

    @ivar purged: True if purge_unreachable has already been called on
    this Rollback, False otherwise.

    @ivar random: A list of random numbers that were generated during the
    execution of this element.
    """
    __version__ = 5
    identifier = None
    not_greedy = False
    checkpointing_suspended = False

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(Rollback, self).__init__()
        self.context = renpy.game.context().rollback_copy()
        self.objects = []
        self.purged = False
        self.random = []
        self.forward = None
        self.stores = {}
        self.delta_ebc = {}
        self.retain_after_load = False
        self.checkpoint = False
        self.hard_checkpoint = False
        self.not_greedy = False
        self.checkpointing_suspended = renpy.game.log.checkpointing_suspended
        global serial
        self.identifier = (generation, serial)
        serial += 1

    def after_upgrade(self, version):
        if False:
            while True:
                i = 10
        if version < 2:
            self.stores = {'store': {}}
            for i in self.store:
                if len(i) == 2:
                    (k, v) = i
                    self.stores['store'][k] = v
                else:
                    (k,) = i
                    self.stores['store'][k] = deleted
        if version < 3:
            self.retain_after_load = False
        if version < 4:
            self.hard_checkpoint = self.checkpoint
        if version < 5:
            self.delta_ebc = {}

    def purge_unreachable(self, reachable, wait):
        if False:
            while True:
                i = 10
        '\n        Adds objects that are reachable from the store of this\n        rollback to the set of reachable objects, and purges\n        information that is stored about totally unreachable objects.\n\n        Returns True if this is the first time this method has been\n        called, or False if it has already been called once before.\n        '
        if self.purged:
            return False
        self.purged = True
        for changes in self.stores.values():
            for (_k, v) in changes.items():
                if v is not deleted:
                    reached(v, reachable, wait)
        reached(self.context.info, reachable, wait)
        reached(self.context.music, reachable, wait)
        reached(self.context.movie, reachable, wait)
        reached(self.context.modes, reachable, wait)
        for d in self.context.dynamic_stack:
            for v in d.values():
                reached(v, reachable, wait)
        reached(self.context.scene_lists.get_all_displayables(), reachable, wait)
        new_objects = []
        objects_changed = True
        seen = set()
        while objects_changed:
            objects_changed = False
            for (o, rb) in self.objects:
                id_o = id(o)
                if id_o in seen or id_o not in reachable:
                    continue
                seen.add(id_o)
                if isinstance(o, NOROLLBACK_TYPES):
                    continue
                objects_changed = True
                new_objects.append((o, rb))
                reached(rb, reachable, wait)
        del self.objects[:]
        self.objects.extend(new_objects)
        return True

    def rollback(self):
        if False:
            while True:
                i = 10
        '\n        Reverts the state of the game to what it was at the start of the\n        previous checkpoint.\n        '
        store_dicts = renpy.python.store_dicts
        for (obj, roll) in reversed(self.objects):
            if roll is not None:
                try:
                    obj._rollback(roll)
                except AttributeError:
                    if not hasattr(obj, '_rollback'):
                        if isinstance(obj, tuple(renpy.config.ex_rollback_classes)):
                            continue
                        elif not renpy.config.developer:
                            continue
                        else:
                            raise Exception('Load or rollback failed because class {} does not inherit from store.object, but did in the past. If this was an intentional change, add the class to config.ex_rollback_classes.'.format(type(obj).__name__))
        for (name, changes) in self.stores.items():
            store = store_dicts.get(name, None)
            if store is None:
                continue
            for (name, value) in changes.items():
                if value is deleted:
                    if name in store:
                        del store[name]
                else:
                    store[name] = value
        for (name, changes) in self.delta_ebc.items():
            store = store_dicts.get(name, None)
            if store is None:
                continue
            store.ever_been_changed -= changes
        rng.pushback(self.random)
        self.rollback_control()

    def rollback_control(self):
        if False:
            print('Hello World!')
        '\n        This rolls back only the control information, while leaving\n        the data information intact.\n        '
        renpy.game.contexts = renpy.game.contexts[:-1] + [self.context]
        renpy.game.log.checkpointing_suspended = self.checkpointing_suspended

class RollbackLog(renpy.object.Object):
    """
    This class manages the list of Rollback objects.

    @ivar log: The log of rollback objects.

    @ivar current: The current rollback object. (Equivalent to
    log[-1])

    @ivar rollback_limit: The number of steps left that we can
    interactively rollback.

    Not serialized:

    @ivar mutated: A dictionary that maps object ids to a tuple of
    (weakref to object, information needed to rollback that object)
    """
    __version__ = 5
    nosave = ['old_store', 'mutated', 'identifier_cache']
    identifier_cache = None
    force_checkpoint = False

    def __init__(self):
        if False:
            while True:
                i = 10
        super(RollbackLog, self).__init__()
        self.log = []
        self.current = None
        self.mutated = {}
        self.rollback_limit = 0
        self.rollback_is_fixed = False
        self.checkpointing_suspended = False
        self.fixed_rollback_boundary = None
        self.forward = []
        self.old_store = {}
        self.rolled_forward = False
        rng.reset()
        self.retain_after_load_flag = False
        self.did_interaction = True
        self.force_checkpoint = False

    def after_setstate(self):
        if False:
            return 10
        self.mutated = {}
        self.rolled_forward = False

    def after_upgrade(self, version):
        if False:
            return 10
        if version < 2:
            self.ever_been_changed = {'store': set(self.ever_been_changed)}
        if version < 3:
            self.rollback_is_fixed = False
            self.fixed_rollback_boundary = None
        if version < 4:
            self.retain_after_load_flag = False
        if version < 5:
            if self.rollback_limit:
                nrbl = 0
                for rb in self.log[-self.rollback_limit:]:
                    if rb.hard_checkpoint:
                        nrbl += 1
                self.rollback_limit = nrbl

    def begin(self, force=False):
        if False:
            while True:
                i = 10
        '\n        Called before a node begins executing, to indicate that the\n        state needs to be saved for rollbacking.\n        '
        self.identifier_cache = None
        context = renpy.game.context()
        if not context.rollback:
            return
        ignore = True
        if force:
            ignore = False
        elif self.did_interaction:
            ignore = False
        elif self.current is not None:
            if self.current.checkpoint:
                ignore = False
            elif self.current.retain_after_load:
                ignore = False
        if ignore:
            return
        self.did_interaction = False
        if self.current is not None:
            self.complete(True)
        else:
            renpy.python.begin_stores()
        while len(self.log) > renpy.config.rollback_length:
            self.log.pop(0)
        if len(self.log) >= 2:
            if self.log[-2].context.current == self.fixed_rollback_boundary:
                self.rollback_is_fixed = False
        if self.rollback_is_fixed and (not self.forward):
            self.fixed_rollback_boundary = self.current.context.current
            self.rollback_is_fixed = False
        self.current = Rollback()
        self.current.retain_after_load = self.retain_after_load_flag
        self.log.append(self.current)
        self.mutated.clear()
        renpy.revertable.mutate_flag = True
        self.rolled_forward = False

    def replace_node(self, old, new):
        if False:
            return 10
        '\n        Replaces references to the `old` ast node with a reference to the\n        `new` ast node.\n        '
        for i in self.log:
            i.context.replace_node(old, new)

    def complete(self, begin=False):
        if False:
            print('Hello World!')
        '\n        Called after a node is finished executing, before a save\n        begins, or right before a rollback is attempted. This may be\n        called more than once between calls to begin, and should always\n        be called after an update to the store but before a rollback\n        occurs.\n\n        `begin`\n            Should be true if called from begin().\n        '
        if self.force_checkpoint:
            self.checkpoint(hard=False)
            self.force_checkpoint = False
        for (name, sd) in renpy.python.store_dicts.items():
            delta = sd.get_changes(begin)
            if delta:
                (self.current.stores[name], self.current.delta_ebc[name]) = delta
        for _i in range(4):
            del self.current.objects[:]
            try:
                for (_k, v) in self.mutated.items():
                    if v is None:
                        continue
                    (ref, clean) = v
                    obj = ref()
                    if obj is None:
                        continue
                    compressed = obj._compress(clean)
                    self.current.objects.append((obj, compressed))
                break
            except RuntimeError:
                pass

    def get_roots(self):
        if False:
            while True:
                i = 10
        '\n        Return a map giving the current roots of the store. This is a\n        map from a variable name in the store to the value of that\n        variable. A variable is only in this map if it has ever been\n        changed since the init phase finished.\n        '
        rv = {}
        for (store_name, sd) in renpy.python.store_dicts.items():
            for name in sd.ever_been_changed:
                if name in sd:
                    rv[store_name + '.' + name] = sd[name]
                else:
                    rv[store_name + '.' + name] = deleted
        for i in reversed(renpy.game.contexts[1:]):
            i.pop_dynamic_roots(rv)
        return rv

    def purge_unreachable(self, roots, wait=None):
        if False:
            i = 10
            return i + 15
        '\n        This is called to purge objects that are unreachable from the\n        roots from the object rollback lists inside the Rollback entries.\n\n        This should be called immediately after complete(), so that there\n        are no changes queued up.\n        '
        global NOROLLBACK_TYPES
        NOROLLBACK_TYPES = (types.ModuleType, renpy.python.StoreModule, SlottedNoRollback, io.IOBase, type)
        reachable = {}
        reached_vars(roots, reachable, wait)
        revlog = self.log[:]
        revlog.reverse()
        for i in revlog:
            if not i.purge_unreachable(reachable, wait):
                break
        reachable.clear()

    def in_rollback(self):
        if False:
            print('Hello World!')
        if self.forward:
            return True
        else:
            return False

    def in_fixed_rollback(self):
        if False:
            i = 10
            return i + 15
        return self.rollback_is_fixed

    def forward_info(self):
        if False:
            return 10
        '\n        Returns the current forward info, if any.\n        '
        if self.forward:
            (name, data) = self.forward[0]
            if self.current.context.current == name:
                return data
        return None

    def checkpoint(self, data=None, keep_rollback=False, hard=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called to indicate that this is a checkpoint, which means\n        that the user may want to rollback to just before this\n        node.\n        '
        if self.checkpointing_suspended:
            hard = False
            self.current.not_greedy = True
        if hard:
            self.retain_after_load_flag = False
        if not renpy.game.context().rollback:
            return
        self.current.checkpoint = True
        if hard and (not self.current.hard_checkpoint):
            if self.rollback_limit < renpy.config.hard_rollback_limit:
                self.rollback_limit += 1
            if hard == 'not_greedy':
                self.current.not_greedy = True
            else:
                self.current.hard_checkpoint = hard
        if self.in_fixed_rollback() and self.forward:
            (fwd_name, fwd_data) = self.forward[0]
            if self.current.context.current == fwd_name:
                self.current.forward = fwd_data
                self.forward.pop(0)
            else:
                self.current.forward = data
                del self.forward[:]
        elif data is not None:
            if self.forward:
                (fwd_name, fwd_data) = self.forward[0]
                if self.current.context.current == fwd_name and data == fwd_data and (keep_rollback or self.rolled_forward):
                    self.forward.pop(0)
                else:
                    del self.forward[:]
            self.current.forward = data

    def suspend_checkpointing(self, flag):
        if False:
            i = 10
            return i + 15
        '\n        Called to temporarily suspend checkpointing, so any rollback\n        will jump to prior to this statement\n        '
        self.checkpointing_suspended = flag
        self.current.not_greedy = True
        renpy.game.contexts[0].force_checkpoint = True

    def block(self, purge=False):
        if False:
            while True:
                i = 10
        '\n        Called to indicate that the user should not be able to rollback\n        through this checkpoint.\n        '
        self.rollback_limit = 0
        if self.current is not None:
            self.current.not_greedy = True
        renpy.game.context().force_checkpoint = True
        if purge:
            del self.log[:]

    def retain_after_load(self):
        if False:
            return 10
        '\n        Called to return data from this statement until the next checkpoint\n        when the game is loaded.\n        '
        if renpy.display.predict.predicting:
            return
        self.retain_after_load_flag = True
        self.current.retain_after_load = True
        for rb in reversed(self.log):
            if rb.hard_checkpoint:
                break
            rb.retain_after_load = True
        renpy.game.context().force_checkpoint = True

    def fix_rollback(self):
        if False:
            return 10
        if not self.rollback_is_fixed and len(self.log) > 1:
            self.fixed_rollback_boundary = self.log[-2].context.current
        renpy.game.context().force_checkpoint = True

    def can_rollback(self):
        if False:
            return 10
        '\n        Returns True if we can rollback.\n        '
        return self.rollback_limit > 0

    def load_failed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is called to try to recover when rollback fails.\n        '
        lfl = renpy.config.load_failed_label
        if callable(lfl):
            lfl = lfl()
        if not lfl:
            raise Exception("Couldn't find a place to stop rolling back. Perhaps the script changed in an incompatible way?")
        rb = self.log.pop()
        rb.rollback()
        while renpy.exports.call_stack_depth():
            renpy.exports.pop_call()
        renpy.game.contexts[0].force_checkpoint = True
        renpy.game.contexts[0].goto_label(lfl)
        raise renpy.game.RestartTopContext()

    def rollback(self, checkpoints, force=False, label=None, greedy=True, on_load=False, abnormal=True, current_label=None):
        if False:
            print('Hello World!')
        "\n        This rolls the system back to the first valid rollback point\n        after having rolled back past the specified number of checkpoints.\n\n        If we're currently executing code, it's expected that complete()\n        will be called before a rollback is attempted.\n\n        force makes us throw an exception if we can't find a place to stop\n        rolling back, otherwise if we run out of log this call has no\n        effect.\n\n        `label`\n            A label that is called after rollback has finished, if the\n            label exists.\n\n        `greedy`\n            If true, rollback will keep going until just after the last\n            checkpoint. If False, it will stop immediately before the\n            current statement.\n\n        `on_load`\n            Should be true if this rollback is being called in response to a\n            load. Used to implement .retain_after_load()\n\n        `abnormal`\n            If true, treats this as an abnormal event, suppressing transitions\n            and so on.\n\n        `current_label`\n            A label that is called when control returns to the current statement,\n            after rollback. (At most one of `current_label` and `label` can be\n            provided.)\n        "
        if checkpoints and self.rollback_limit <= 0 and (not force):
            return
        self.purge_unreachable(self.get_roots())
        revlog = []
        while self.log:
            rb = self.log.pop()
            revlog.append(rb)
            if rb.hard_checkpoint:
                self.rollback_limit -= 1
            if rb.hard_checkpoint or (on_load and rb.checkpoint):
                checkpoints -= 1
            if checkpoints <= 0:
                if renpy.game.script.has_label(rb.context.current):
                    break
        else:
            revlog.reverse()
            self.log.extend(revlog)
            if force:
                self.load_failed()
            else:
                print("Can't find a place to rollback to. Not rolling back.")
            return
        force_checkpoint = False
        while greedy and self.log:
            rb = self.log[-1]
            if not renpy.game.script.has_label(rb.context.current):
                break
            if rb.hard_checkpoint:
                break
            if rb.not_greedy:
                break
            revlog.append(self.log.pop())
        old_contexts = list(renpy.game.contexts)
        try:
            if renpy.game.context().rollback:
                replace_context = False
                other_contexts = []
            else:
                replace_context = True
                other_contexts = renpy.game.contexts[1:]
                renpy.game.contexts = renpy.game.contexts[0:1]
            if on_load and revlog[-1].retain_after_load:
                retained = revlog.pop()
                self.retain_after_load_flag = True
            else:
                retained = None
            come_from = None
            if current_label is not None:
                come_from = renpy.game.context().current
                label = current_label
            for rb in revlog:
                rb.rollback()
                if rb.context.current == self.fixed_rollback_boundary and rb.context.current:
                    self.rollback_is_fixed = True
                if rb.forward is not None:
                    self.forward.insert(0, (rb.context.current, rb.forward))
            if retained is not None:
                retained.rollback_control()
                self.log.append(retained)
        except Exception:
            renpy.game.contexts = old_contexts
            raise
        if label is not None and come_from is None:
            come_from = renpy.game.context().current
        if come_from is not None:
            renpy.game.context().come_from(come_from, label)
        renpy.game.interface.suppress_transition = abnormal
        if force:
            rng.reset()
            del self.forward[:]
        renpy.game.after_rollback = abnormal
        renpy.audio.audio.rollback()
        for i in renpy.game.contexts:
            i.scene_lists.remove_all_hidden()
        renpy.game.contexts.extend(other_contexts)
        renpy.exports.execute_default_statement(False)
        self.mutated.clear()
        renpy.python.begin_stores()
        if replace_context:
            if force_checkpoint:
                renpy.game.contexts[0].force_checkpoint = True
            self.current = Rollback()
            self.current.context = renpy.game.contexts[0].rollback_copy()
            if self.log is not None:
                self.log.append(self.current)
            raise renpy.game.RestartTopContext()
        else:
            self.current = Rollback()
            self.current.context = renpy.game.context().rollback_copy()
            if self.log is not None:
                self.log.append(self.current)
            if force_checkpoint:
                renpy.game.context().force_checkpoint = True
            raise renpy.game.RestartContext()

    def freeze(self, wait=None):
        if False:
            i = 10
            return i + 15
        '\n        This is called to freeze the store and the log, in preparation\n        for serialization. The next call on log should either be\n        unfreeze (called after a serialization reload) or discard_freeze()\n        (called after the save is complete).\n        '
        self.complete(False)
        roots = self.get_roots()
        self.purge_unreachable(roots, wait=wait)
        self.current.purged = False
        return roots

    def discard_freeze(self):
        if False:
            i = 10
            return i + 15
        '\n        Called to indicate that we will not be restoring from the\n        frozen state.\n        '

    def unfreeze(self, roots, label=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Used to unfreeze the game state after a load of this log\n        object. This call will always throw an exception. If we're\n        lucky, it's the one that indicates load was successful.\n\n        @param roots: The roots returned from freeze.\n\n        @param label: The label that is jumped to in the game script\n        after rollback has finished, if it exists.\n        "
        renpy.display.screen.before_restart()
        renpy.game.log = self
        renpy.python.clean_stores()
        renpy.translation.init_translation()
        store_dicts = renpy.python.store_dicts
        for (name, value) in roots.items():
            if '.' in name:
                (store_name, name) = name.rsplit('.', 1)
            else:
                store_name = 'store'
            if store_name not in store_dicts:
                continue
            store = store_dicts[store_name]
            store.ever_been_changed.add(name)
            if value is deleted:
                if name in store:
                    del store[name]
            else:
                store[name] = value
        greedy = getattr(renpy.store, '_greedy_rollback', True)
        greedy = renpy.session.pop('_greedy_rollback', greedy)
        self.rollback(0, force=True, label=label, greedy=greedy, on_load=True)

    def build_identifier_cache(self):
        if False:
            print('Hello World!')
        if self.identifier_cache is not None:
            return
        rollback_limit = self.rollback_limit
        checkpoints = 1
        self.identifier_cache = {}
        for i in reversed(self.log):
            if i.identifier is not None:
                if renpy.game.script.has_label(i.context.current):
                    self.identifier_cache[i.identifier] = checkpoints
            if i.hard_checkpoint:
                checkpoints += 1
            if i.checkpoint:
                rollback_limit -= 1
            if not rollback_limit:
                break

    def get_identifier_checkpoints(self, identifier):
        if False:
            i = 10
            return i + 15
        self.build_identifier_cache()
        return self.identifier_cache.get(identifier, None)