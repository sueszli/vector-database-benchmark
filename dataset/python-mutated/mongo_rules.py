import os
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
from uses_services.scripts.is_mongo_running import __file__ as is_mongo_running_full_path
from uses_services.target_types import UsesServicesField

@dataclass(frozen=True)
class UsesMongoRequest:
    """One or more targets need a running mongo service using these settings.

    The db_* attributes represent the db connection settings from st2.conf.
    In st2 code, they come from:
        oslo_config.cfg.CONF.database.{host,port,db_name,connection_timeout}
    """
    db_host: str = '127.0.0.1'
    db_port: int = 27017
    db_name: str = f"st2-test{os.environ.get('ST2TESTS_PARALLEL_SLOT', '')}"
    db_connection_timeout: int = 3000

@dataclass(frozen=True)
class MongoIsRunning:
    pass

class PytestUsesMongoRequest(PytestPluginSetupRequest):

    @classmethod
    def is_applicable(cls, target: Target) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not target.has_field(UsesServicesField):
            return False
        uses = target.get(UsesServicesField).value
        return uses is not None and 'mongo' in uses

@rule(desc='Ensure mongodb is running and accessible before running tests.', level=LogLevel.DEBUG)
async def mongo_is_running_for_pytest(request: PytestUsesMongoRequest) -> PytestPluginSetup:
    _ = await Get(MongoIsRunning, UsesMongoRequest())
    return PytestPluginSetup()

@rule(desc='Test to see if mongodb is running and accessible.', level=LogLevel.DEBUG)
async def mongo_is_running(request: UsesMongoRequest, platform: Platform) -> MongoIsRunning:
    script_path = './is_mongo_running.py'
    with open(is_mongo_running_full_path, 'rb') as script_file:
        script_contents = script_file.read()
    (script_digest, mongoengine_pex) = await MultiGet(Get(Digest, CreateDigest([FileContent(script_path, script_contents)])), Get(VenvPex, PexRequest(output_filename='mongoengine.pex', internal_only=True, requirements=PexRequirements({'mongoengine', 'pymongo'}))))
    result = await Get(FallibleProcessResult, VenvPexProcess(mongoengine_pex, argv=(script_path, request.db_host, str(request.db_port), request.db_name, str(request.db_connection_timeout)), input_digest=script_digest, description='Checking to see if Mongo is up and accessible.', cache_scope=ProcessCacheScope.PER_SESSION, level=LogLevel.DEBUG))
    is_running = result.exit_code == 0
    if is_running:
        return MongoIsRunning()
    raise ServiceMissingError.generate(platform=platform, messages=ServiceSpecificMessages(service='mongo', service_start_cmd_el_7='service mongo start', service_start_cmd_el='systemctl start mongod', not_installed_clause_el='this is one way to install it:', install_instructions_el=dedent('                # Add key and repo for the latest stable MongoDB (4.0)\n                sudo rpm --import https://www.mongodb.org/static/pgp/server-4.0.asc\n                sudo sh -c "cat <<EOT > /etc/yum.repos.d/mongodb-org-4.repo\n                [mongodb-org-4]\n                name=MongoDB Repository\n                baseurl=https://repo.mongodb.org/yum/redhat/${OSRELEASE_VERSION}/mongodb-org/4.0/x86_64/\n                gpgcheck=1\n                enabled=1\n                gpgkey=https://www.mongodb.org/static/pgp/server-4.0.asc\n                EOT"\n                # Install mongo\n                sudo yum -y install mongodb-org\n                # Don\'t forget to start mongo.\n                '), service_start_cmd_deb='systemctl start mongod', not_installed_clause_deb='this is one way to install it:', install_instructions_deb=dedent("                sudo apt-get install -y mongodb-org\n                # Don't forget to start mongo.\n                "), service_start_cmd_generic='systemctl start mongod'))

def rules():
    if False:
        while True:
            i = 10
    return [*collect_rules(), UnionRule(PytestPluginSetupRequest, PytestUsesMongoRequest), *pex_rules()]