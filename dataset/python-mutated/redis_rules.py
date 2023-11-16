from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from pants.backend.python.goals.pytest_runner import PytestPluginSetupRequest, PytestPluginSetup
from pants.backend.python.util_rules.pex import PexRequest, PexRequirements, VenvPex, VenvPexProcess, rules as pex_rules
from pants.engine.fs import CreateDigest, Digest, FileContent
from pants.engine.rules import collect_rules, Get, MultiGet, rule
from pants.engine.process import FallibleProcessResult, ProcessCacheScope
from pants.engine.target import Target
from pants.engine.unions import UnionRule
from pants.util.logging import LogLevel
from uses_services.exceptions import ServiceMissingError, ServiceSpecificMessages
from uses_services.platform_rules import Platform
from uses_services.scripts.is_redis_running import __file__ as is_redis_running_full_path
from uses_services.target_types import UsesServicesField

@dataclass(frozen=True)
class UsesRedisRequest:
    """One or more targets need a running redis service using these settings.

    The coord_* attributes represent the coordination settings from st2.conf.
    In st2 code, they come from:
        oslo_config.cfg.CONF.coordination.url
    """
    coord_url: str = 'redis://127.0.0.1:6379'

@dataclass(frozen=True)
class RedisIsRunning:
    pass

class PytestUsesRedisRequest(PytestPluginSetupRequest):

    @classmethod
    def is_applicable(cls, target: Target) -> bool:
        if False:
            while True:
                i = 10
        if not target.has_field(UsesServicesField):
            return False
        uses = target.get(UsesServicesField).value
        return uses is not None and 'redis' in uses

@rule(desc='Ensure redis is running and accessible before running tests.', level=LogLevel.DEBUG)
async def redis_is_running_for_pytest(request: PytestUsesRedisRequest) -> PytestPluginSetup:
    _ = await Get(RedisIsRunning, UsesRedisRequest())
    return PytestPluginSetup()

@rule(desc='Test to see if redis is running and accessible.', level=LogLevel.DEBUG)
async def redis_is_running(request: UsesRedisRequest, platform: Platform) -> RedisIsRunning:
    script_path = './is_redis_running.py'
    with open(is_redis_running_full_path, 'rb') as script_file:
        script_contents = script_file.read()
    (script_digest, tooz_pex) = await MultiGet(Get(Digest, CreateDigest([FileContent(script_path, script_contents)])), Get(VenvPex, PexRequest(output_filename='tooz.pex', internal_only=True, requirements=PexRequirements({'tooz', 'redis'}))))
    result = await Get(FallibleProcessResult, VenvPexProcess(tooz_pex, argv=(script_path, request.coord_url), input_digest=script_digest, description='Checking to see if Redis is up and accessible.', cache_scope=ProcessCacheScope.PER_SESSION, level=LogLevel.DEBUG))
    is_running = result.exit_code == 0
    if is_running:
        return RedisIsRunning()
    raise ServiceMissingError.generate(platform=platform, messages=ServiceSpecificMessages(service='redis', service_start_cmd_el_7='service redis start', service_start_cmd_el='systemctl start redis', not_installed_clause_el='this is one way to install it:', install_instructions_el=dedent("                sudo yum -y install redis\n                # Don't forget to start redis.\n                "), service_start_cmd_deb='systemctl start redis', not_installed_clause_deb='this is one way to install it:', install_instructions_deb=dedent("                sudo apt-get install -y mongodb redis\n                # Don't forget to start redis.\n                "), service_start_cmd_generic='systemctl start redis'))

def rules():
    if False:
        print('Hello World!')
    return [*collect_rules(), UnionRule(PytestPluginSetupRequest, PytestUsesRedisRequest), *pex_rules()]