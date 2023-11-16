import os
from dataclasses import dataclass
from pants.engine.addresses import Address
from pants.engine.fs import GlobMatchErrorBehavior, PathGlobs, Paths
from pants.engine.rules import Get, collect_rules, MultiGet, rule, UnionRule
from pants.engine.target import DependenciesRequest, ExplicitlyProvidedDependencies, FieldSet, InferDependenciesRequest, InferredDependencies
from pants.util.logging import LogLevel
from pack_metadata.target_types import PacksGlobDependencies

@dataclass(frozen=True)
class PacksGlobInferenceFieldSet(FieldSet):
    required_fields = (PacksGlobDependencies,)
    dependencies: PacksGlobDependencies

class InferPacksGlobDependencies(InferDependenciesRequest):
    infer_from = PacksGlobInferenceFieldSet

@rule(desc='Inferring packs glob dependencies', level=LogLevel.DEBUG)
async def infer_packs_globs_dependencies(request: InferPacksGlobDependencies) -> InferredDependencies:
    address = request.field_set.address
    (pack_build_paths, explicitly_provided_deps) = await MultiGet(Get(Paths, PathGlobs([os.path.join(address.spec_path, '*', 'BUILD')], glob_match_error_behavior=GlobMatchErrorBehavior.error, description_of_origin=f"{address}'s packs glob")), Get(ExplicitlyProvidedDependencies, DependenciesRequest(request.field_set.dependencies)))
    implicit_packs_deps = {Address(os.path.dirname(path)) for path in pack_build_paths.files}
    inferred_packs_deps = implicit_packs_deps - explicitly_provided_deps.ignores - explicitly_provided_deps.includes
    return InferredDependencies(inferred_packs_deps)

def rules():
    if False:
        i = 10
        return i + 15
    return [*collect_rules(), UnionRule(InferDependenciesRequest, InferPacksGlobDependencies)]