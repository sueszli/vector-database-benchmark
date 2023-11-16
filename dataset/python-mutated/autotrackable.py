"""Dependency tracking for trackable objects."""
import warnings
from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.types import core as core_types
from tensorflow.python.util.tf_export import tf_export

@tf_export('__internal__.tracking.AutoTrackable', v1=[])
class AutoTrackable(base.Trackable):
    """Manages dependencies on other objects.

  `Trackable` objects may have dependencies: other `Trackable` objects
  which should be saved if the object declaring the dependency is saved. A
  correctly saveable program has a dependency graph such that if changing a
  global variable affects an object (e.g. changes the behavior of any of its
  methods) then there is a chain of dependencies from the influenced object to
  the variable.

  Dependency edges have names, and are created implicitly when a
  `Trackable` object is assigned to an attribute of another
  `Trackable` object. For example:

  ```
  obj = Trackable()
  obj.v = ResourceVariable(0.)
  ```

  The `Trackable` object `obj` now has a dependency named "v" on a
  variable.

  `Trackable` objects may specify `Tensor`s to be saved and restored
  directly (e.g. a `Variable` indicating how to save itself) rather than through
  dependencies on other objects. See
  `Trackable._gather_saveables_for_checkpoint` for details.
  """

    def __setattr__(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        'Support self.foo = trackable syntax.'
        try:
            if getattr(self, name) is value:
                return
        except AttributeError:
            pass
        if getattr(self, '_self_setattr_tracking', True):
            value = data_structures.sticky_attribute_assignment(trackable=self, value=value, name=name)
        super(AutoTrackable, self).__setattr__(name, value)

    def __delattr__(self, name):
        if False:
            while True:
                i = 10
        self._delete_tracking(name)
        super(AutoTrackable, self).__delattr__(name)

    def _no_dependency(self, value):
        if False:
            i = 10
            return i + 15
        'Override to allow TrackableBase to disable dependency tracking.'
        return data_structures.NoDependency(value)

    def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
        if False:
            while True:
                i = 10
        'Returns all children of a trackable, including functions.'
        if save_type != base.SaveType.SAVEDMODEL:
            return super(AutoTrackable, self)._trackable_children(save_type, **kwargs)
        functions = {}
        try:
            logging_verbosity = logging.get_verbosity()
            logging.set_verbosity(logging.FATAL)
            for attribute_name in dir(self):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        attribute_value = getattr(self, attribute_name, None)
                except Exception:
                    attribute_value = None
                if isinstance(attribute_value, (def_function.Function, defun.ConcreteFunction)):
                    functions[attribute_name] = attribute_value
        finally:
            logging.set_verbosity(logging_verbosity)
        for fn in functions.values():
            if isinstance(fn, def_function.Function):
                fn._list_all_concrete_functions_for_serialization()
        children = {}
        for (name, child) in self._checkpoint_dependencies:
            if isinstance(child, (core_types.PolymorphicFunction, core_types.ConcreteFunction)):
                continue
            if name in functions and child is not functions[name]:
                raise ValueError(f"Can't save object because it has multiple children with the same name. Object: {self}, attribute name: {name}, child 1: {child}, child 2: {functions[name]}")
            children[name] = child
        children.update(functions)
        return children

    def _delete_tracking(self, name):
        if False:
            i = 10
            return i + 15
        'Removes the tracking of name.'
        self._maybe_initialize_trackable()
        if name in self._unconditional_dependency_names:
            del self._unconditional_dependency_names[name]
            for (index, (dep_name, _)) in enumerate(self._unconditional_checkpoint_dependencies):
                if dep_name == name:
                    del self._unconditional_checkpoint_dependencies[index]
                    break

    def _add_trackable_child(self, name, value):
        if False:
            while True:
                i = 10
        self.__setattr__(name, value)