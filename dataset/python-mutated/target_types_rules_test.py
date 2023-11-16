from __future__ import annotations
from textwrap import dedent
from pants.backend.python.target_types import PythonSourceTarget, PythonSourcesGeneratorTarget
from pants.backend.python.target_types_rules import rules as python_target_types_rules
from pants.engine.addresses import Address
from pants.engine.target import InferredDependencies
from pants.testutil.rule_runner import QueryRule, RuleRunner
from .target_types_rules import InferPacksGlobDependencies, PacksGlobInferenceFieldSet, rules as pack_metadata_target_types_rules
from .target_types import PacksGlob

def test_infer_packs_globs_dependencies() -> None:
    if False:
        while True:
            i = 10
    rule_runner = RuleRunner(rules=[*python_target_types_rules(), *pack_metadata_target_types_rules(), QueryRule(InferredDependencies, (InferPacksGlobDependencies,))], target_types=[PythonSourceTarget, PythonSourcesGeneratorTarget, PacksGlob])
    rule_runner.write_files({'packs/BUILD': dedent('                python_sources(\n                    name="git_submodule",\n                    sources=["./git_submodule/*.py"],\n                )\n\n                packs_glob(\n                    name="all_packs_glob",\n                    dependencies=[\n                        "!./configs",  # explicit ignore\n                        "./a",         # explicit include\n                    ],\n                )\n                '), 'packs/a/BUILD': 'python_sources()', 'packs/a/__init__.py': '', 'packs/a/fixture.py': '', 'packs/b/BUILD': dedent('                python_sources(\n                    dependencies=["packs/configs/b.yaml"],\n                )\n                '), 'packs/b/__init__.py': '', 'packs/b/fixture.py': '', 'packs/c/BUILD': 'python_sources()', 'packs/c/__init__.py': '', 'packs/c/fixture.py': '', 'packs/d/BUILD': 'python_sources()', 'packs/d/__init__.py': '', 'packs/d/fixture.py': '', 'packs/git_submodule/__init__.py': '', 'packs/git_submodule/fixture.py': '', 'packs/configs/BUILD': dedent('                resources(\n                    sources=["*.yaml"],\n                )\n                '), 'packs/configs/b.yaml': dedent('                ---\n                # pack config for pack b\n                ')})

    def run_dep_inference(address: Address) -> InferredDependencies:
        if False:
            for i in range(10):
                print('nop')
        args = ['--source-root-patterns=/packs']
        rule_runner.set_options(args, env_inherit={'PATH', 'PYENV_ROOT', 'HOME'})
        target = rule_runner.get_target(address)
        return rule_runner.request(InferredDependencies, [InferPacksGlobDependencies(PacksGlobInferenceFieldSet.create(target))])
    assert run_dep_inference(Address('packs', target_name='all_packs_glob')) == InferredDependencies([Address('packs/b'), Address('packs/c'), Address('packs/d')])