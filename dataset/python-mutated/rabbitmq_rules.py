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
from uses_services.scripts.is_rabbitmq_running import __file__ as is_rabbitmq_running_full_path
from uses_services.target_types import UsesServicesField

@dataclass(frozen=True)
class UsesRabbitMQRequest:
    """One or more targets need a running rabbitmq service using these settings.

    The mq_* attributes represent the messaging settings from st2.conf.
    In st2 code, they come from:
        oslo_config.cfg.CONF.messaging.{url,cluster_urls}
    """
    mq_urls: tuple[str] = ('amqp://guest:guest@127.0.0.1:5672//',)

@dataclass(frozen=True)
class RabbitMQIsRunning:
    pass

class PytestUsesRabbitMQRequest(PytestPluginSetupRequest):

    @classmethod
    def is_applicable(cls, target: Target) -> bool:
        if False:
            while True:
                i = 10
        if not target.has_field(UsesServicesField):
            return False
        uses = target.get(UsesServicesField).value
        return uses is not None and 'rabbitmq' in uses

@rule(desc='Ensure rabbitmq is running and accessible before running tests.', level=LogLevel.DEBUG)
async def rabbitmq_is_running_for_pytest(request: PytestUsesRabbitMQRequest) -> PytestPluginSetup:
    _ = await Get(RabbitMQIsRunning, UsesRabbitMQRequest())
    return PytestPluginSetup()

@rule(desc='Test to see if rabbitmq is running and accessible.', level=LogLevel.DEBUG)
async def rabbitmq_is_running(request: UsesRabbitMQRequest, platform: Platform) -> RabbitMQIsRunning:
    script_path = './is_rabbitmq_running.py'
    with open(is_rabbitmq_running_full_path, 'rb') as script_file:
        script_contents = script_file.read()
    (script_digest, kombu_pex) = await MultiGet(Get(Digest, CreateDigest([FileContent(script_path, script_contents)])), Get(VenvPex, PexRequest(output_filename='kombu.pex', internal_only=True, requirements=PexRequirements({'kombu'}))))
    result = await Get(FallibleProcessResult, VenvPexProcess(kombu_pex, argv=(script_path, *request.mq_urls), input_digest=script_digest, description='Checking to see if RabbitMQ is up and accessible.', cache_scope=ProcessCacheScope.PER_SESSION, level=LogLevel.DEBUG))
    is_running = result.exit_code == 0
    if is_running:
        return RabbitMQIsRunning()
    raise ServiceMissingError.generate(platform=platform, messages=ServiceSpecificMessages(service='rabbitmq', service_start_cmd_el_7='service rabbitmq-server start', service_start_cmd_el='systemctl start rabbitmq-server', not_installed_clause_el='this is one way to install it:', install_instructions_el=dedent('                # Add key and repo for erlang and RabbitMQ\n                curl -sL https://packagecloud.io/install/repositories/rabbitmq/erlang/script.rpm.sh | sudo bash\n                curl -sL https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.rpm.sh | sudo bash\n                sudo yum makecache -y --disablerepo=\'*\' --enablerepo=\'rabbitmq_rabbitmq-server\'\n                # Check for any required version constraints in our docs:\n                # https://docs.stackstorm.com/latest/install/rhel{platform.distro_major_version}.html\n\n                # Install erlang and RabbitMQ (and possibly constrain the version)\n                sudo yum -y install erlang{\'\' if platform.distro_major_version == "7" else \'-*\'}\n                sudo yum -y install rabbitmq-server\n                # Don\'t forget to start rabbitmq-server.\n                '), service_start_cmd_deb='systemctl start rabbitmq-server', not_installed_clause_deb='try the quick start script here:', install_instructions_deb=dedent('                https://www.rabbitmq.com/install-debian.html#apt-cloudsmith\n                '), service_start_cmd_generic='systemctl start rabbitmq-server'))

def rules():
    if False:
        while True:
            i = 10
    return [*collect_rules(), UnionRule(PytestPluginSetupRequest, PytestUsesRabbitMQRequest), *pex_rules()]