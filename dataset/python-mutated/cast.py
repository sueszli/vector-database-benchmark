from b2.build import targets, virtual_target, property_set, type as type_
from b2.manager import get_manager
from b2.util import bjam_signature, is_iterable_typed

class CastTargetClass(targets.TypedTarget):

    def construct(self, name, source_targets, ps):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(name, basestring)
        assert is_iterable_typed(source_targets, virtual_target.VirtualTarget)
        assert isinstance(ps, property_set.PropertySet)
        result = []
        for s in source_targets:
            if not isinstance(s, virtual_target.FileTarget):
                get_manager().errors()("Source to the 'cast' metatager is not a file")
            if s.action():
                get_manager().errors()("Only non-derived targets allowed as sources for 'cast'.")
            r = s.clone_with_different_type(self.type())
            result.append(get_manager().virtual_targets().register(r))
        return (property_set.empty(), result)

@bjam_signature((['name', 'type'], ['sources', '*'], ['requirements', '*'], ['default_build', '*'], ['usage_requirements', '*']))
def cast(name, type, sources, requirements, default_build, usage_requirements):
    if False:
        i = 10
        return i + 15
    from b2.manager import get_manager
    t = get_manager().targets()
    project = get_manager().projects().current()
    real_type = type_.type_from_rule_name(type)
    if not real_type:
        real_type = type
    return t.main_target_alternative(CastTargetClass(name, project, real_type, t.main_target_sources(sources, name), t.main_target_requirements(requirements, project), t.main_target_default_build(default_build, project), t.main_target_usage_requirements(usage_requirements, project)))
get_manager().projects().add_rule('cast', cast)