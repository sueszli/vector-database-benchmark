from typing import List
from sentry.mediators.mediator import Mediator
from sentry.mediators.param import Param
from sentry.plugins.base import plugins
from sentry.services.hybrid_cloud.integration.model import RpcIntegration
from sentry.services.hybrid_cloud.organization.model import RpcOrganization
from sentry.services.hybrid_cloud.repository import repository_service
from sentry.services.hybrid_cloud.repository.model import RpcRepository
from sentry.utils.cache import memoize

class Migrator(Mediator):
    integration = Param(RpcIntegration)
    organization = Param(RpcOrganization)
    using = None

    def call(self):
        if False:
            print('Hello World!')
        for project in self.projects:
            for plugin in plugins.for_project(project):
                if plugin.slug != self.integration.provider:
                    continue
                if self.all_repos_migrated(plugin.slug):
                    self.disable_for_all_projects(plugin)

    def all_repos_migrated(self, provider):
        if False:
            i = 10
            return i + 15
        return all((r.integration_id is not None for r in self.repos_for_provider(provider)))

    def disable_for_all_projects(self, plugin):
        if False:
            while True:
                i = 10
        for project in self.projects:
            try:
                self.log(at='disable', project=project.slug, plugin=plugin.slug)
                plugin.disable(project=project)
            except NotImplementedError:
                pass

    def repos_for_provider(self, provider):
        if False:
            return 10
        return [r for r in self.repositories if r.provider == provider]

    @property
    def repositories(self) -> List[RpcRepository]:
        if False:
            while True:
                i = 10
        return repository_service.get_repositories(organization_id=self.organization.id)

    @memoize
    def projects(self):
        if False:
            print('Hello World!')
        return list(self.organization.projects)

    @property
    def plugins(self):
        if False:
            return 10
        return [plugins.configurable_for_project(project) for project in self.projects]

    @property
    def _logging_context(self):
        if False:
            return 10
        return {'org': self.organization.slug, 'integration_id': self.integration.id, 'integration_provider': self.integration.provider}