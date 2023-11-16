import os.path
from distutils import log
from .base import ROOT, BaseBuildCommand

class BuildIntegrationDocsCommand(BaseBuildCommand):
    description = 'build integration docs'

    def get_dist_paths(self):
        if False:
            while True:
                i = 10
        return [os.path.join(ROOT, 'src', 'sentry', 'integration-docs')]

    def _build(self):
        if False:
            i = 10
            return i + 15
        from sentry.utils.integrationdocs import sync_docs
        log.info('downloading integration docs')
        sync_docs()