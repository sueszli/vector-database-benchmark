import hashlib
import bjam
from b2.util.utility import *
import property, feature
import b2.build.feature
from b2.exceptions import *
from b2.build.property import get_abbreviated_paths
from b2.util.sequence import unique
from b2.util.set import difference
from b2.util import cached, abbreviate_dashed, is_iterable_typed
from b2.manager import get_manager

def reset():
    if False:
        print('Hello World!')
    ' Clear the module state. This is mainly for testing purposes.\n    '
    global __cache
    __cache = {}
reset()

def create(raw_properties=[]):
    if False:
        for i in range(10):
            print('nop')
    " Creates a new 'PropertySet' instance for the given raw properties,\n        or returns an already existing one.\n    "
    assert is_iterable_typed(raw_properties, property.Property) or is_iterable_typed(raw_properties, basestring)
    if len(raw_properties) > 0 and isinstance(raw_properties[0], property.Property):
        x = raw_properties
    else:
        x = [property.create_from_string(ps) for ps in raw_properties]
    x = sorted(set(x), key=lambda p: (p.feature.name, p.value, p.condition))
    key = tuple((p.id for p in x))
    if key not in __cache:
        __cache[key] = PropertySet(x)
    return __cache[key]

def create_with_validation(raw_properties):
    if False:
        i = 10
        return i + 15
    " Creates new 'PropertySet' instances after checking\n        that all properties are valid and converting implicit\n        properties into gristed form.\n    "
    assert is_iterable_typed(raw_properties, basestring)
    properties = [property.create_from_string(s) for s in raw_properties]
    property.validate(properties)
    return create(properties)

def empty():
    if False:
        while True:
            i = 10
    ' Returns PropertySet with empty set of properties.\n    '
    return create()

def create_from_user_input(raw_properties, jamfile_module, location):
    if False:
        while True:
            i = 10
    "Creates a property-set from the input given by the user, in the\n    context of 'jamfile-module' at 'location'"
    assert is_iterable_typed(raw_properties, basestring)
    assert isinstance(jamfile_module, basestring)
    assert isinstance(location, basestring)
    properties = property.create_from_strings(raw_properties, True)
    properties = property.translate_paths(properties, location)
    properties = property.translate_indirect(properties, jamfile_module)
    project_id = get_manager().projects().attributeDefault(jamfile_module, 'id', None)
    if not project_id:
        project_id = os.path.abspath(location)
    properties = property.translate_dependencies(properties, project_id, location)
    properties = property.expand_subfeatures_in_conditions(properties)
    return create(properties)

def refine_from_user_input(parent_requirements, specification, jamfile_module, location):
    if False:
        while True:
            i = 10
    'Refines requirements with requirements provided by the user.\n    Specially handles "-<property>value" syntax in specification\n     to remove given requirements.\n     - parent-requirements -- property-set object with requirements\n       to refine\n     - specification -- string list of requirements provided by the use\n     - project-module -- the module to which context indirect features\n       will be bound.\n     - location -- the path to which path features are relative.'
    assert isinstance(parent_requirements, PropertySet)
    assert is_iterable_typed(specification, basestring)
    assert isinstance(jamfile_module, basestring)
    assert isinstance(location, basestring)
    if not specification:
        return parent_requirements
    add_requirements = []
    remove_requirements = []
    for r in specification:
        if r[0] == '-':
            remove_requirements.append(r[1:])
        else:
            add_requirements.append(r)
    if remove_requirements:
        ps = create_from_user_input(remove_requirements, jamfile_module, location)
        parent_requirements = create(difference(parent_requirements.all(), ps.all()))
        specification = add_requirements
    requirements = create_from_user_input(specification, jamfile_module, location)
    return parent_requirements.refine(requirements)

class PropertySet:
    """ Class for storing a set of properties.
        - there's 1<->1 correspondence between identity and value. No
          two instances of the class are equal. To maintain this property,
          the 'PropertySet.create' rule should be used to create new instances.
          Instances are immutable.

        - each property is classified with regard to it's effect on build
          results. Incidental properties have no effect on build results, from
          Boost.Build point of view. Others are either free, or non-free, which we
          call 'base'. Each property belong to exactly one of those categories and
          it's possible to get list of properties in each category.

          In addition, it's possible to get list of properties with specific
          attribute.

        - several operations, like and refine and as_path are provided. They all use
          caching whenever possible.
    """

    def __init__(self, properties=None):
        if False:
            while True:
                i = 10
        if properties is None:
            properties = []
        assert is_iterable_typed(properties, property.Property)
        self.all_ = properties
        self._all_set = {p.id for p in properties}
        self.incidental_ = []
        self.free_ = []
        self.base_ = []
        self.dependency_ = []
        self.non_dependency_ = []
        self.conditional_ = []
        self.non_conditional_ = []
        self.propagated_ = []
        self.link_incompatible = []
        self.refined_ = {}
        self.added_ = {}
        self.defaults_ = None
        self.expanded_ = None
        self.composites_ = None
        self.subfeatures_ = None
        self.propagated_ps_ = None
        self.feature_map_ = None
        self.target_path_ = None
        self.as_path_ = None
        self.evaluated_ = {}
        self.lazy_properties = []
        for p in properties:
            f = p.feature
            if isinstance(p, property.LazyProperty):
                self.lazy_properties.append(p)
            elif f.incidental:
                self.incidental_.append(p)
            elif f.free:
                self.free_.append(p)
            else:
                self.base_.append(p)
            if p.condition:
                self.conditional_.append(p)
            else:
                self.non_conditional_.append(p)
            if f.dependency:
                self.dependency_.append(p)
            elif not isinstance(p, property.LazyProperty):
                self.non_dependency_.append(p)
            if f.propagated:
                self.propagated_.append(p)
            if f.link_incompatible:
                self.link_incompatible.append(p)

    def all(self):
        if False:
            return 10
        return self.all_

    def raw(self):
        if False:
            return 10
        ' Returns the list of stored properties.\n        '
        return [p._to_raw for p in self.all_]

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return ' '.join((p._to_raw for p in self.all_))

    def base(self):
        if False:
            print('Hello World!')
        ' Returns properties that are neither incidental nor free.\n        '
        result = [p for p in self.lazy_properties if not (p.feature.incidental or p.feature.free)]
        result.extend(self.base_)
        return result

    def free(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns free properties which are not dependency properties.\n        '
        result = [p for p in self.lazy_properties if not p.feature.incidental and p.feature.free]
        result.extend(self.free_)
        return result

    def non_free(self):
        if False:
            for i in range(10):
                print('nop')
        return self.base() + self.incidental()

    def dependency(self):
        if False:
            return 10
        ' Returns dependency properties.\n        '
        result = [p for p in self.lazy_properties if p.feature.dependency]
        result.extend(self.dependency_)
        return self.dependency_

    def non_dependency(self):
        if False:
            while True:
                i = 10
        ' Returns properties that are not dependencies.\n        '
        result = [p for p in self.lazy_properties if not p.feature.dependency]
        result.extend(self.non_dependency_)
        return result

    def conditional(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns conditional properties.\n        '
        return self.conditional_

    def non_conditional(self):
        if False:
            i = 10
            return i + 15
        ' Returns properties that are not conditional.\n        '
        return self.non_conditional_

    def incidental(self):
        if False:
            print('Hello World!')
        ' Returns incidental properties.\n        '
        result = [p for p in self.lazy_properties if p.feature.incidental]
        result.extend(self.incidental_)
        return result

    def refine(self, requirements):
        if False:
            while True:
                i = 10
        " Refines this set's properties using the requirements passed as an argument.\n        "
        assert isinstance(requirements, PropertySet)
        if requirements not in self.refined_:
            r = property.refine(self.all_, requirements.all_)
            self.refined_[requirements] = create(r)
        return self.refined_[requirements]

    def expand(self):
        if False:
            i = 10
            return i + 15
        if not self.expanded_:
            expanded = feature.expand(self.all_)
            self.expanded_ = create(expanded)
        return self.expanded_

    def expand_subfeatures(self):
        if False:
            return 10
        if not self.subfeatures_:
            self.subfeatures_ = create(feature.expand_subfeatures(self.all_))
        return self.subfeatures_

    def evaluate_conditionals(self, context=None):
        if False:
            print('Hello World!')
        assert isinstance(context, (PropertySet, type(None)))
        if not context:
            context = self
        if context not in self.evaluated_:
            self.evaluated_[context] = create(property.evaluate_conditionals_in_context(self.all(), context))
        return self.evaluated_[context]

    def propagated(self):
        if False:
            return 10
        if not self.propagated_ps_:
            self.propagated_ps_ = create(self.propagated_)
        return self.propagated_ps_

    def add_defaults(self):
        if False:
            while True:
                i = 10
        if not self.defaults_:
            expanded = feature.add_defaults(self.all_)
            self.defaults_ = create(expanded)
        return self.defaults_

    def as_path(self):
        if False:
            print('Hello World!')
        if not self.as_path_:

            def path_order(p1, p2):
                if False:
                    return 10
                i1 = p1.feature.implicit
                i2 = p2.feature.implicit
                if i1 != i2:
                    return i2 - i1
                else:
                    return cmp(p1.feature.name, p2.feature.name)
            properties = feature.minimize(self.base_)
            properties.sort(path_order)
            components = []
            for p in properties:
                f = p.feature
                if f.implicit:
                    components.append(p.value)
                else:
                    value = f.name.replace(':', '-') + '-' + p.value
                    if property.get_abbreviated_paths():
                        value = abbreviate_dashed(value)
                    components.append(value)
            self.as_path_ = '/'.join(components)
        return self.as_path_

    def target_path(self):
        if False:
            for i in range(10):
                print('nop')
        " Computes the target path that should be used for\n            target with these properties.\n            Returns a tuple of\n              - the computed path\n              - if the path is relative to build directory, a value of\n                'true'.\n        "
        if not self.target_path_:
            l = self.get('<location>')
            if l:
                computed = l[0]
                is_relative = False
            else:
                p = self.as_path()
                if hash_maybe:
                    p = hash_maybe(p)
                prefix = self.get('<location-prefix>')
                if prefix:
                    if len(prefix) > 1:
                        raise AlreadyDefined("Two <location-prefix> properties specified: '%s'" % prefix)
                    computed = os.path.join(prefix[0], p)
                else:
                    computed = p
                if not computed:
                    computed = '.'
                is_relative = True
            self.target_path_ = (computed, is_relative)
        return self.target_path_

    def add(self, ps):
        if False:
            print('Hello World!')
        ' Creates a new property set containing the properties in this one,\n            plus the ones of the property set passed as argument.\n        '
        assert isinstance(ps, PropertySet)
        if ps not in self.added_:
            self.added_[ps] = create(self.all_ + ps.all())
        return self.added_[ps]

    def add_raw(self, properties):
        if False:
            return 10
        ' Creates a new property set containing the properties in this one,\n            plus the ones passed as argument.\n        '
        return self.add(create(properties))

    def get(self, feature):
        if False:
            print('Hello World!')
        " Returns all values of 'feature'.\n        "
        if type(feature) == type([]):
            feature = feature[0]
        if not isinstance(feature, b2.build.feature.Feature):
            feature = b2.build.feature.get(feature)
        assert isinstance(feature, b2.build.feature.Feature)
        if self.feature_map_ is None:
            self.feature_map_ = {}
            for v in self.all_:
                if v.feature not in self.feature_map_:
                    self.feature_map_[v.feature] = []
                self.feature_map_[v.feature].append(v.value)
        return self.feature_map_.get(feature, [])

    @cached
    def get_properties(self, feature):
        if False:
            print('Hello World!')
        "Returns all contained properties associated with 'feature'"
        if not isinstance(feature, b2.build.feature.Feature):
            feature = b2.build.feature.get(feature)
        assert isinstance(feature, b2.build.feature.Feature)
        result = []
        for p in self.all_:
            if p.feature == feature:
                result.append(p)
        return result

    def __contains__(self, item):
        if False:
            print('Hello World!')
        return item.id in self._all_set

def hash(p):
    if False:
        print('Hello World!')
    m = hashlib.md5()
    m.update(p)
    return m.hexdigest()
hash_maybe = hash if '--hash' in bjam.variable('ARGV') else None