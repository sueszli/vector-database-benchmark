import json
from pants.backend.python.util_rules.pex import PexRequest, PexRequirements, VenvPex, VenvPexProcess, rules as pex_rules
from pants.engine.fs import CreateDigest, Digest, FileContent
from pants.engine.process import ProcessCacheScope, ProcessResult
from pants.engine.rules import collect_rules, Get, MultiGet, rule
from pants.util.logging import LogLevel
from uses_services.scripts.inspect_platform import Platform, __file__ as inspect_platform_full_path
__all__ = ['Platform', 'get_platform', 'rules']

@rule(desc='Get details (os, distro, etc) about platform running tests.', level=LogLevel.DEBUG)
async def get_platform() -> Platform:
    script_path = './inspect_platform.py'
    with open(inspect_platform_full_path, 'rb') as script_file:
        script_contents = script_file.read()
    (script_digest, distro_pex) = await MultiGet(Get(Digest, CreateDigest([FileContent(script_path, script_contents)])), Get(VenvPex, PexRequest(output_filename='distro.pex', internal_only=True, requirements=PexRequirements({'distro'}))))
    result = await Get(ProcessResult, VenvPexProcess(distro_pex, argv=(script_path,), input_digest=script_digest, description='Introspecting platform (arch, os, distro)', cache_scope=ProcessCacheScope.PER_RESTART_SUCCESSFUL, level=LogLevel.DEBUG))
    platform = json.loads(result.stdout)
    return Platform(**platform)

def rules():
    if False:
        for i in range(10):
            print('nop')
    return [*collect_rules(), *pex_rules()]