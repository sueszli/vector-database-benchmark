import bjam
import re
import os.path
import string
import types
from b2.util import path, utility, set, is_iterable_typed
from b2.util.utility import add_grist, get_grist, ungrist, replace_grist, get_value
from b2.util.sequence import unique
from b2.tools import common
from b2.exceptions import *
import b2.build.type
import b2.build.property_set as property_set
import b2.build.property as property
from b2.manager import get_manager
from b2.util import bjam_signature
__re_starts_with_at = re.compile('^@(.*)')

class VirtualTargetRegistry:

    def __init__(self, manager):
        if False:
            return 10
        self.manager_ = manager
        self.files_ = {}
        self.cache_ = {}
        self.actual_ = {}
        self.recent_targets_ = []
        self.all_targets_ = []
        self.next_id_ = 0

    def register(self, target):
        if False:
            i = 10
            return i + 15
        " Registers a new virtual target. Checks if there's already registered target, with the same\n            name, type, project and subvariant properties, and also with the same sources\n            and equal action. If such target is found it is retured and 'target' is not registered.\n            Otherwise, 'target' is registered and returned.\n        "
        assert isinstance(target, VirtualTarget)
        if target.path():
            signature = target.path() + '-' + target.name()
        else:
            signature = '-' + target.name()
        result = None
        if signature not in self.cache_:
            self.cache_[signature] = []
        for t in self.cache_[signature]:
            a1 = t.action()
            a2 = target.action()
            if not result:
                if not a1 and (not a2):
                    result = t
                elif a1 and a2 and (a1.action_name() == a2.action_name()) and (a1.sources() == a2.sources()):
                    ps1 = a1.properties()
                    ps2 = a2.properties()
                    p1 = ps1.base() + ps1.free() + b2.util.set.difference(ps1.dependency(), ps1.incidental())
                    p2 = ps2.base() + ps2.free() + b2.util.set.difference(ps2.dependency(), ps2.incidental())
                    if p1 == p2:
                        result = t
        if not result:
            self.cache_[signature].append(target)
            result = target
        self.recent_targets_.append(result)
        self.all_targets_.append(result)
        return result

    def from_file(self, file, file_location, project):
        if False:
            while True:
                i = 10
        " Creates a virtual target with appropriate name and type from 'file'.\n            If a target with that name in that project was already created, returns that already\n            created target.\n            TODO: more correct way would be to compute path to the file, based on name and source location\n            for the project, and use that path to determine if the target was already created.\n            TODO: passing project with all virtual targets starts to be annoying.\n        "
        if __debug__:
            from .targets import ProjectTarget
            assert isinstance(file, basestring)
            assert isinstance(file_location, basestring)
            assert isinstance(project, ProjectTarget)
        path = os.path.join(os.getcwd(), file_location, file)
        path = os.path.normpath(path)
        if path in self.files_:
            return self.files_[path]
        file_type = b2.build.type.type(file)
        result = FileTarget(file, file_type, project, None, file_location)
        self.files_[path] = result
        return result

    def recent_targets(self):
        if False:
            print('Hello World!')
        "Each target returned by 'register' is added to a list of\n        'recent-target', returned by this function. So, this allows\n        us to find all targets created when building a given main\n        target, even if the target."
        return self.recent_targets_

    def clear_recent_targets(self):
        if False:
            while True:
                i = 10
        self.recent_targets_ = []

    def all_targets(self):
        if False:
            while True:
                i = 10
        return self.all_targets_

    def select_by_type(self, type, targets):
        if False:
            while True:
                i = 10
        return [t for t in targets if b2.build.type.is_sybtype(t.type(), type)]

    def register_actual_name(self, actual_name, virtual_target):
        if False:
            return 10
        assert isinstance(actual_name, basestring)
        assert isinstance(virtual_target, VirtualTarget)
        if actual_name in self.actual_:
            cs1 = self.actual_[actual_name].creating_subvariant()
            cs2 = virtual_target.creating_subvariant()
            cmt1 = cs1.main_target()
            cmt2 = cs2.main_target()
            action1 = self.actual_[actual_name].action()
            action2 = virtual_target.action()
            properties_added = []
            properties_removed = []
            if action1 and action2:
                p1 = action1.properties()
                p1 = p1.raw()
                p2 = action2.properties()
                p2 = p2.raw()
                properties_removed = set.difference(p1, p2)
                if not properties_removed:
                    properties_removed = ['none']
                properties_added = set.difference(p2, p1)
                if not properties_added:
                    properties_added = ['none']
            get_manager().errors()("Duplicate name of actual target: '%s'\nprevious virtual target '%s'\ncreated from '%s'\nanother virtual target '%s'\ncreated from '%s'\nadded properties:\n%s\nremoved properties:\n%s\n" % (actual_name, self.actual_[actual_name], cmt1.project().location(), virtual_target, cmt2.project().location(), '\n'.join(('\t' + p for p in properties_added)), '\n'.join(('\t' + p for p in properties_removed))))
        else:
            self.actual_[actual_name] = virtual_target

    def add_suffix(self, specified_name, file_type, prop_set):
        if False:
            while True:
                i = 10
        " Appends the suffix appropriate to 'type/property_set' combination\n            to the specified name and returns the result.\n        "
        assert isinstance(specified_name, basestring)
        assert isinstance(file_type, basestring)
        assert isinstance(prop_set, property_set.PropertySet)
        suffix = b2.build.type.generated_target_suffix(file_type, prop_set)
        if suffix:
            return specified_name + '.' + suffix
        else:
            return specified_name

class VirtualTarget:
    """ Potential target. It can be converted into jam target and used in
        building, if needed. However, it can be also dropped, which allows
        to search for different transformation and select only one.
        name:    name of this target.
        project: project to which this target belongs.
    """

    def __init__(self, name, project):
        if False:
            return 10
        if __debug__:
            from .targets import ProjectTarget
            assert isinstance(name, basestring)
            assert isinstance(project, ProjectTarget)
        self.name_ = name
        self.project_ = project
        self.dependencies_ = []
        self.always_ = False
        self.made_ = {}

    def manager(self):
        if False:
            print('Hello World!')
        return self.project_.manager()

    def virtual_targets(self):
        if False:
            print('Hello World!')
        return self.manager().virtual_targets()

    def name(self):
        if False:
            while True:
                i = 10
        ' Name of this target.\n        '
        return self.name_

    def project(self):
        if False:
            i = 10
            return i + 15
        ' Project of this target.\n        '
        return self.project_

    def depends(self, d):
        if False:
            print('Hello World!')
        " Adds additional instances of 'VirtualTarget' that this\n            one depends on.\n        "
        self.dependencies_ = unique(self.dependencies_ + d).sort()

    def dependencies(self):
        if False:
            return 10
        return self.dependencies_

    def always(self):
        if False:
            for i in range(10):
                print('nop')
        self.always_ = True

    def actualize(self, scanner=None):
        if False:
            while True:
                i = 10
        " Generates all the actual targets and sets up build actions for\n            this target.\n\n            If 'scanner' is specified, creates an additional target\n            with the same location as actual target, which will depend on the\n            actual target and be associated with 'scanner'. That additional\n            target is returned. See the docs (#dependency_scanning) for rationale.\n            Target must correspond to a file if 'scanner' is specified.\n\n            If scanner is not specified, then actual target is returned.\n        "
        if __debug__:
            from .scanner import Scanner
            assert scanner is None or isinstance(scanner, Scanner)
        actual_name = self.actualize_no_scanner()
        if self.always_:
            bjam.call('ALWAYS', actual_name)
        if not scanner:
            return actual_name
        else:
            g = '-'.join([ungrist(get_grist(actual_name)), str(id(scanner))])
            name = replace_grist(actual_name, '<' + g + '>')
            if name not in self.made_:
                self.made_[name] = True
                self.project_.manager().engine().add_dependency(name, actual_name)
                self.actualize_location(name)
                self.project_.manager().scanners().install(scanner, name, str(self))
            return name

    def actualize_action(self, target):
        if False:
            return 10
        " Sets up build actions for 'target'. Should call appropriate rules\n            and set target variables.\n        "
        raise BaseException('method should be defined in derived classes')

    def actualize_location(self, target):
        if False:
            while True:
                i = 10
        " Sets up variables on 'target' which specify its location.\n        "
        raise BaseException('method should be defined in derived classes')

    def path(self):
        if False:
            while True:
                i = 10
        ' If the target is generated one, returns the path where it will be\n            generated. Otherwise, returns empty list.\n        '
        raise BaseException('method should be defined in derived classes')

    def actual_name(self):
        if False:
            while True:
                i = 10
        ' Return that actual target name that should be used\n            (for the case where no scanner is involved)\n        '
        raise BaseException('method should be defined in derived classes')

class AbstractFileTarget(VirtualTarget):
    """ Target which correspond to a file. The exact mapping for file
        is not yet specified in this class. (TODO: Actually, the class name
        could be better...)

        May be a source file (when no action is specified), or
        derived file (otherwise).

        The target's grist is concatenation of project's location,
        properties of action (for derived files), and, optionally,
        value identifying the main target.

        exact:  If non-empty, the name is exactly the name
                created file should have. Otherwise, the '__init__'
                method will add suffix obtained from 'type' by
                calling 'type.generated-target-suffix'.

        type:   optional type of this target.
    """

    def __init__(self, name, type, project, action=None, exact=False):
        if False:
            print('Hello World!')
        assert isinstance(type, basestring) or type is None
        assert action is None or isinstance(action, Action)
        assert isinstance(exact, (int, bool))
        VirtualTarget.__init__(self, name, project)
        self.type_ = type
        self.action_ = action
        self.exact_ = exact
        if action:
            action.add_targets([self])
            if self.type and (not exact):
                self.__adjust_name(name)
        self.actual_name_ = None
        self.path_ = None
        self.intermediate_ = False
        self.creating_subvariant_ = None
        self.root_ = False

    def type(self):
        if False:
            i = 10
            return i + 15
        return self.type_

    def set_path(self, path):
        if False:
            while True:
                i = 10
        ' Sets the path. When generating target name, it will override any path\n            computation from properties.\n        '
        assert isinstance(path, basestring)
        self.path_ = os.path.normpath(path)

    def action(self):
        if False:
            print('Hello World!')
        ' Returns the action.\n        '
        return self.action_

    def root(self, set=None):
        if False:
            while True:
                i = 10
        " Sets/gets the 'root' flag. Target is root is it directly correspods to some\n            variant of a main target.\n        "
        assert isinstance(set, (int, bool, type(None)))
        if set:
            self.root_ = True
        return self.root_

    def creating_subvariant(self, s=None):
        if False:
            return 10
        " Gets or sets the subvariant which created this target. Subvariant\n        is set when target is brought into existance, and is never changed\n        after that. In particual, if target is shared by subvariant, only\n        the first is stored.\n        s:  If specified, specified the value to set,\n                which should be instance of 'subvariant' class.\n        "
        assert s is None or isinstance(s, Subvariant)
        if s and (not self.creating_subvariant()):
            if self.creating_subvariant():
                raise BaseException("Attempt to change 'dg'")
            else:
                self.creating_subvariant_ = s
        return self.creating_subvariant_

    def actualize_action(self, target):
        if False:
            print('Hello World!')
        assert isinstance(target, basestring)
        if self.action_:
            self.action_.actualize()

    def str(self):
        if False:
            while True:
                i = 10
        a = self.action()
        name_dot_type = self.name_ + '.' + self.type_
        if a:
            action_name = a.action_name()
            ss = [s.str() for s in a.sources()]
            return '{ %s-%s %s}' % (action_name, name_dot_type, str(ss))
        else:
            return '{ ' + name_dot_type + ' }'

    def actual_name(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.actual_name_:
            self.actual_name_ = '<' + self.grist() + '>' + os.path.normpath(self.name_)
        return self.actual_name_

    def grist(self):
        if False:
            return 10
        "Helper to 'actual_name', above. Compute unique prefix used to distinguish\n            this target from other targets with the same name which create different\n            file.\n        "
        path = self.path()
        if path:
            return 'p' + path
        else:
            project_location = self.project_.get('location')
            path_components = b2.util.path.split(project_location)
            location_grist = '!'.join(path_components)
            if self.action_:
                ps = self.action_.properties()
                property_grist = ps.as_path()
                if property_grist:
                    location_grist = location_grist + '/' + property_grist
            return 'l' + location_grist

    def __adjust_name(self, specified_name):
        if False:
            return 10
        'Given the target name specified in constructor, returns the\n        name which should be really used, by looking at the <tag> properties.\n        The tag properties come in two flavour:\n          - <tag>value,\n          - <tag>@rule-name\n        In the first case, value is just added to name\n        In the second case, the specified rule is called with specified name,\n        target type and properties and should return the new name.\n        If not <tag> property is specified, or the rule specified by\n        <tag> returns nothing, returns the result of calling\n        virtual-target.add-suffix'
        assert isinstance(specified_name, basestring)
        if self.action_:
            ps = self.action_.properties()
        else:
            ps = property_set.empty()
        tag = ps.get('<tag>')
        if tag:
            if len(tag) > 1:
                get_manager().errors()('<tag>@rulename is present but is not the only <tag> feature')
            tag = tag[0]
            if callable(tag):
                self.name_ = tag(specified_name, self.type_, ps)
            else:
                if not tag[0] == '@':
                    self.manager_.errors()("The value of the <tag> feature must be '@rule-nane'")
                exported_ps = b2.util.value_to_jam(ps, methods=True)
                self.name_ = b2.util.call_jam_function(tag[1:], specified_name, self.type_, exported_ps)
                if self.name_:
                    self.name_ = self.name_[0]
        if not tag or not self.name_:
            self.name_ = add_prefix_and_suffix(specified_name, self.type_, ps)

    def actualize_no_scanner(self):
        if False:
            i = 10
            return i + 15
        name = self.actual_name()
        if not self.made_:
            self.made_[name] = True
            if self.action_:
                self.virtual_targets().register_actual_name(name, self)
            for i in self.dependencies_:
                self.manager_.engine().add_dependency(name, i.actualize())
            self.actualize_location(name)
            self.actualize_action(name)
        return name

@bjam_signature((['specified_name'], ['type'], ['property_set']))
def add_prefix_and_suffix(specified_name, type, property_set):
    if False:
        for i in range(10):
            print('nop')
    "Appends the suffix appropriate to 'type/property-set' combination\n    to the specified name and returns the result."
    property_set = b2.util.jam_to_value_maybe(property_set)
    suffix = ''
    if type:
        suffix = b2.build.type.generated_target_suffix(type, property_set)
    if get_grist(suffix):
        suffix = ungrist(suffix)
    elif suffix:
        suffix = '.' + suffix
    prefix = ''
    if type:
        prefix = b2.build.type.generated_target_prefix(type, property_set)
    if specified_name.startswith(prefix):
        prefix = ''
    if not prefix:
        prefix = ''
    if not suffix:
        suffix = ''
    return prefix + specified_name + suffix

class FileTarget(AbstractFileTarget):
    """ File target with explicitly known location.

        The file path is determined as
           - value passed to the 'set_path' method, if any
           - for derived files, project's build dir, joined with components
             that describe action's properties. If the free properties
             are not equal to the project's reference properties
             an element with name of main target is added.
           - for source files, project's source dir

        The file suffix is
            - the value passed to the 'suffix' method, if any, or
            - the suffix which correspond to the target's type.
    """

    def __init__(self, name, type, project, action=None, path=None, exact=False):
        if False:
            return 10
        assert isinstance(type, basestring) or type is None
        assert action is None or isinstance(action, Action)
        assert isinstance(exact, (int, bool))
        AbstractFileTarget.__init__(self, name, type, project, action, exact)
        self.path_ = path

    def __str__(self):
        if False:
            return 10
        if self.type_:
            return self.name_ + '.' + self.type_
        else:
            return self.name_

    def clone_with_different_type(self, new_type):
        if False:
            return 10
        assert isinstance(new_type, basestring)
        return FileTarget(self.name_, new_type, self.project_, self.action_, self.path_, exact=True)

    def actualize_location(self, target):
        if False:
            while True:
                i = 10
        assert isinstance(target, basestring)
        engine = self.project_.manager_.engine()
        if self.action_:
            path = self.path()
            engine.set_target_variable(target, 'LOCATE', path)
            engine.add_dependency(target, path)
            common.mkdir(engine, path)
            d = os.path.dirname(get_value(target))
            if d:
                d = os.path.join(path, d)
                engine.add_dependency(target, d)
                common.mkdir(engine, d)
            engine.add_dependency('<e>%s' % get_value(target), target)
            engine.add_dependency('<e>%s' % os.path.join(path, get_value(target)), target)
        else:
            engine.set_target_variable(target, 'SEARCH', self.project_.get('source-location'))

    def path(self):
        if False:
            i = 10
            return i + 15
        ' Returns the directory for this target.\n        '
        if not self.path_:
            if self.action_:
                p = self.action_.properties()
                (target_path, relative_to_build_dir) = p.target_path()
                if relative_to_build_dir:
                    target_path = os.path.join(self.project_.build_dir(), target_path)
                self.path_ = target_path
        return os.path.normpath(self.path_)

class NotFileTarget(AbstractFileTarget):

    def __init__(self, name, project, action):
        if False:
            i = 10
            return i + 15
        assert isinstance(action, Action)
        AbstractFileTarget.__init__(self, name, None, project, action)

    def path(self):
        if False:
            i = 10
            return i + 15
        'Returns nothing, to indicate that target path is not known.'
        return None

    def actualize_location(self, target):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(target, basestring)
        bjam.call('NOTFILE', target)
        bjam.call('ALWAYS', target)
        bjam.call('NOUPDATE', target)

class Action:
    """ Class which represents an action.
        Both 'targets' and 'sources' should list instances of 'VirtualTarget'.
        Action name should name a rule with this prototype
            rule action_name ( targets + : sources * : properties * )
        Targets and sources are passed as actual jam targets. The rule may
        not establish dependency relationship, but should do everything else.
    """

    def __init__(self, manager, sources, action_name, prop_set):
        if False:
            return 10
        assert is_iterable_typed(sources, VirtualTarget)
        assert isinstance(action_name, basestring) or action_name is None
        assert isinstance(prop_set, property_set.PropertySet)
        self.sources_ = sources
        self.action_name_ = action_name
        if not prop_set:
            prop_set = property_set.empty()
        self.properties_ = prop_set
        if not all((isinstance(v, VirtualTarget) for v in prop_set.get('implicit-dependency'))):
            import pdb
            pdb.set_trace()
        self.manager_ = manager
        self.engine_ = self.manager_.engine()
        self.targets_ = []
        self.actualized_ = False
        self.dependency_only_sources_ = []
        self.actual_sources_ = []

    def add_targets(self, targets):
        if False:
            i = 10
            return i + 15
        assert is_iterable_typed(targets, VirtualTarget)
        self.targets_ += targets

    def replace_targets(self, old_targets, new_targets):
        if False:
            while True:
                i = 10
        assert is_iterable_typed(old_targets, VirtualTarget)
        assert is_iterable_typed(new_targets, VirtualTarget)
        self.targets_ = [t for t in self.targets_ if not t in old_targets] + new_targets

    def targets(self):
        if False:
            for i in range(10):
                print('nop')
        return self.targets_

    def sources(self):
        if False:
            i = 10
            return i + 15
        return self.sources_

    def action_name(self):
        if False:
            print('Hello World!')
        return self.action_name_

    def properties(self):
        if False:
            print('Hello World!')
        return self.properties_

    def actualize(self):
        if False:
            for i in range(10):
                print('nop')
        ' Generates actual build instructions.\n        '
        if self.actualized_:
            return
        self.actualized_ = True
        ps = self.properties()
        properties = self.adjust_properties(ps)
        actual_targets = []
        for i in self.targets():
            actual_targets.append(i.actualize())
        self.actualize_sources(self.sources(), properties)
        self.engine_.add_dependency(actual_targets, self.actual_sources_ + self.dependency_only_sources_)
        import toolset
        toolset.set_target_variables(self.manager_, self.action_name_, actual_targets, properties)
        engine = self.manager_.engine()
        bjam.call('set-target-variable', actual_targets, '.action', repr(self))
        self.manager_.engine().set_update_action(self.action_name_, actual_targets, self.actual_sources_, properties)
        self.manager_.engine().set_update_action('common.Clean', 'clean-all', actual_targets)
        return actual_targets

    def actualize_source_type(self, sources, prop_set):
        if False:
            while True:
                i = 10
        " Helper for 'actualize_sources'.\n            For each passed source, actualizes it with the appropriate scanner.\n            Returns the actualized virtual targets.\n        "
        assert is_iterable_typed(sources, VirtualTarget)
        assert isinstance(prop_set, property_set.PropertySet)
        result = []
        for i in sources:
            scanner = None
            if i.type():
                scanner = b2.build.type.get_scanner(i.type(), prop_set)
            r = i.actualize(scanner)
            result.append(r)
        return result

    def actualize_sources(self, sources, prop_set):
        if False:
            for i in range(10):
                print('nop')
        " Creates actual jam targets for sources. Initializes two member\n            variables:\n            'self.actual_sources_' -- sources which are passed to updating action\n            'self.dependency_only_sources_' -- sources which are made dependencies, but\n            are not used otherwise.\n\n            New values will be *appended* to the variables. They may be non-empty,\n            if caller wants it.\n        "
        assert is_iterable_typed(sources, VirtualTarget)
        assert isinstance(prop_set, property_set.PropertySet)
        dependencies = self.properties_.get('<dependency>')
        self.dependency_only_sources_ += self.actualize_source_type(dependencies, prop_set)
        self.actual_sources_ += self.actualize_source_type(sources, prop_set)
        implicit = self.properties_.get('<implicit-dependency>')
        for i in implicit:
            i.actualize()

    def adjust_properties(self, prop_set):
        if False:
            return 10
        " Determines real properties when trying building with 'properties'.\n            This is last chance to fix properties, for example to adjust includes\n            to get generated headers correctly. Default implementation returns\n            its argument.\n        "
        assert isinstance(prop_set, property_set.PropertySet)
        return prop_set

class NullAction(Action):
    """ Action class which does nothing --- it produces the targets with
        specific properties out of nowhere. It's needed to distinguish virtual
        targets with different properties that are known to exist, and have no
        actions which create them.
    """

    def __init__(self, manager, prop_set):
        if False:
            i = 10
            return i + 15
        assert isinstance(prop_set, property_set.PropertySet)
        Action.__init__(self, manager, [], None, prop_set)

    def actualize(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.actualized_:
            self.actualized_ = True
            for i in self.targets():
                i.actualize()

class NonScanningAction(Action):
    """Class which acts exactly like 'action', except that the sources
    are not scanned for dependencies."""

    def __init__(self, sources, action_name, property_set):
        if False:
            print('Hello World!')
        Action.__init__(self, b2.manager.get_manager(), sources, action_name, property_set)

    def actualize_source_type(self, sources, ps=None):
        if False:
            print('Hello World!')
        assert is_iterable_typed(sources, VirtualTarget)
        assert isinstance(ps, property_set.PropertySet) or ps is None
        result = []
        for s in sources:
            result.append(s.actualize())
        return result

def traverse(target, include_roots=False, include_sources=False):
    if False:
        while True:
            i = 10
    " Traverses the dependency graph of 'target' and return all targets that will\n        be created before this one is created. If root of some dependency graph is\n        found during traversal, it's either included or not, dependencing of the\n        value of 'include_roots'. In either case, sources of root are not traversed.\n    "
    assert isinstance(target, VirtualTarget)
    assert isinstance(include_roots, (int, bool))
    assert isinstance(include_sources, (int, bool))
    result = []
    if target.action():
        action = target.action()
        result += action.targets()
        for t in action.sources():
            if not t.root():
                result += traverse(t, include_roots, include_sources)
            elif include_roots:
                result.append(t)
    elif include_sources:
        result.append(target)
    return result

def clone_action(action, new_project, new_action_name, new_properties):
    if False:
        i = 10
        return i + 15
    "Takes an 'action' instances and creates new instance of it\n    and all produced target. The rule-name and properties are set\n    to 'new-rule-name' and 'new-properties', if those are specified.\n    Returns the cloned action."
    if __debug__:
        from .targets import ProjectTarget
        assert isinstance(action, Action)
        assert isinstance(new_project, ProjectTarget)
        assert isinstance(new_action_name, basestring)
        assert isinstance(new_properties, property_set.PropertySet)
    if not new_action_name:
        new_action_name = action.action_name()
    if not new_properties:
        new_properties = action.properties()
    cloned_action = action.__class__(action.manager_, action.sources(), new_action_name, new_properties)
    cloned_targets = []
    for target in action.targets():
        n = target.name()
        cloned_target = FileTarget(n, target.type(), new_project, cloned_action, exact=True)
        d = target.dependencies()
        if d:
            cloned_target.depends(d)
        cloned_target.root(target.root())
        cloned_target.creating_subvariant(target.creating_subvariant())
        cloned_targets.append(cloned_target)
    return cloned_action

class Subvariant:

    def __init__(self, main_target, prop_set, sources, build_properties, sources_usage_requirements, created_targets):
        if False:
            return 10
        '\n        main_target:                 The instance of MainTarget class\n        prop_set:                    Properties requested for this target\n        sources:\n        build_properties:            Actually used properties\n        sources_usage_requirements:  Properties propagated from sources\n        created_targets:             Top-level created targets\n        '
        if __debug__:
            from .targets import AbstractTarget
            assert isinstance(main_target, AbstractTarget)
            assert isinstance(prop_set, property_set.PropertySet)
            assert is_iterable_typed(sources, VirtualTarget)
            assert isinstance(build_properties, property_set.PropertySet)
            assert isinstance(sources_usage_requirements, property_set.PropertySet)
            assert is_iterable_typed(created_targets, VirtualTarget)
        self.main_target_ = main_target
        self.properties_ = prop_set
        self.sources_ = sources
        self.build_properties_ = build_properties
        self.sources_usage_requirements_ = sources_usage_requirements
        self.created_targets_ = created_targets
        self.usage_requirements_ = None
        deps = build_properties.get('<implicit-dependency>')
        self.other_dg_ = []
        for d in deps:
            self.other_dg_.append(d.creating_subvariant())
        self.other_dg_ = unique(self.other_dg_)
        self.implicit_includes_cache_ = {}
        self.target_directories_ = None

    def main_target(self):
        if False:
            i = 10
            return i + 15
        return self.main_target_

    def created_targets(self):
        if False:
            for i in range(10):
                print('nop')
        return self.created_targets_

    def requested_properties(self):
        if False:
            while True:
                i = 10
        return self.properties_

    def build_properties(self):
        if False:
            while True:
                i = 10
        return self.build_properties_

    def sources_usage_requirements(self):
        if False:
            i = 10
            return i + 15
        return self.sources_usage_requirements_

    def set_usage_requirements(self, usage_requirements):
        if False:
            i = 10
            return i + 15
        assert isinstance(usage_requirements, property_set.PropertySet)
        self.usage_requirements_ = usage_requirements

    def usage_requirements(self):
        if False:
            return 10
        return self.usage_requirements_

    def all_referenced_targets(self, result):
        if False:
            while True:
                i = 10
        'Returns all targets referenced by this subvariant,\n        either directly or indirectly, and either as sources,\n        or as dependency properties. Targets referred with\n        dependency property are returned a properties, not targets.'
        if __debug__:
            from .property import Property
            assert is_iterable_typed(result, (VirtualTarget, Property))
        deps = self.build_properties().dependency()
        all_targets = self.sources_ + deps
        r = []
        for e in all_targets:
            if not e in result:
                result.add(e)
                if isinstance(e, property.Property):
                    t = e.value
                else:
                    t = e
                cs = t.creating_subvariant()
                if cs:
                    r.append(cs)
        r = unique(r)
        for s in r:
            if s != self:
                s.all_referenced_targets(result)

    def implicit_includes(self, feature, target_type):
        if False:
            while True:
                i = 10
        " Returns the properties which specify implicit include paths to\n            generated headers. This traverses all targets in this subvariant,\n            and subvariants referred by <implcit-dependecy>properties.\n            For all targets which are of type 'target-type' (or for all targets,\n            if 'target_type' is not specified), the result will contain\n            <$(feature)>path-to-that-target.\n        "
        assert isinstance(feature, basestring)
        assert isinstance(target_type, basestring)
        if not target_type:
            key = feature
        else:
            key = feature + '-' + target_type
        result = self.implicit_includes_cache_.get(key)
        if not result:
            target_paths = self.all_target_directories(target_type)
            target_paths = unique(target_paths)
            result = ['<%s>%s' % (feature, p) for p in target_paths]
            self.implicit_includes_cache_[key] = result
        return result

    def all_target_directories(self, target_type=None):
        if False:
            return 10
        assert isinstance(target_type, (basestring, type(None)))
        if not self.target_directories_:
            self.target_directories_ = self.compute_target_directories(target_type)
        return self.target_directories_

    def compute_target_directories(self, target_type=None):
        if False:
            while True:
                i = 10
        assert isinstance(target_type, (basestring, type(None)))
        result = []
        for t in self.created_targets():
            if not target_type or b2.build.type.is_derived(t.type(), target_type):
                result.append(t.path())
        for d in self.other_dg_:
            result.extend(d.all_target_directories(target_type))
        result = unique(result)
        return result