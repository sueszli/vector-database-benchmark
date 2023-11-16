import os
import shutil
from unittest import mock
from django.conf import settings
from readthedocs.api.v2.serializers import VersionAdminSerializer
from readthedocs.builds.constants import BUILD_STATE_TRIGGERED
from readthedocs.projects.constants import MKDOCS

class BuildEnvironmentMocker:

    def __init__(self, project, version, build, requestsmock):
        if False:
            i = 10
            return i + 15
        self.project = project
        self.version = version
        self.build = build
        self.requestsmock = requestsmock
        self.patches = {}
        self.mocks = {}

    def start(self):
        if False:
            while True:
                i = 10
        self._mock_api()
        self._mock_environment()
        self._mock_git_repository()
        self._mock_artifact_builders()
        self._mock_storage()
        for (k, p) in self.patches.items():
            self.mocks[k] = p.start()

    def stop(self):
        if False:
            i = 10
            return i + 15
        for (k, m) in self.patches.items():
            m.stop()

    def add_file_in_repo_checkout(self, path, content):
        if False:
            for i in range(10):
                print('nop')
        '\n        A quick way to emulate that a file is in the repo.\n\n        Does not change git data.\n        '
        destination = os.path.join(self.project_repository_path, path)
        open(destination, 'w').write(content)
        return destination

    def _mock_artifact_builders(self):
        if False:
            while True:
                i = 10
        self.patches['builder.pdf.PdfBuilder.pdf_file_name'] = mock.patch('readthedocs.doc_builder.backends.sphinx.PdfBuilder.pdf_file_name', 'project-slug.pdf')
        self.patches['builder.pdf.LatexBuildCommand.run'] = mock.patch('readthedocs.doc_builder.backends.sphinx.LatexBuildCommand.run', return_value=mock.MagicMock(output='stdout', successful=True))
        self.patches['builder.pdf.glob'] = mock.patch('readthedocs.doc_builder.backends.sphinx.glob', return_value=['output.file'])
        self.patches['builder.pdf.os.path.getmtime'] = mock.patch('readthedocs.doc_builder.backends.sphinx.os.path.getmtime', return_value=1)
        self.patches['environment.run_command_class'] = mock.patch('readthedocs.projects.tasks.builds.LocalBuildEnvironment.run_command_class', return_value=mock.MagicMock(output='stdout', successful=True))
        self.patches['builder.html.mkdocs.MkdocsHTML.append_conf'] = mock.patch('readthedocs.doc_builder.backends.mkdocs.MkdocsHTML.append_conf')
        self.patches['builder.html.mkdocs.MkdocsHTML.get_final_doctype'] = mock.patch('readthedocs.doc_builder.backends.mkdocs.MkdocsHTML.get_final_doctype', return_value=MKDOCS)
        self.patches['builder.html.sphinx.HtmlBuilder.append_conf'] = mock.patch('readthedocs.doc_builder.backends.sphinx.HtmlBuilder.append_conf')

    def _mock_git_repository(self):
        if False:
            while True:
                i = 10
        self.patches['git.Backend.run'] = mock.patch('readthedocs.vcs_support.backends.git.Backend.run', return_value=(0, 'stdout', 'stderr'))
        self._counter = 0
        self.project_repository_path = '/tmp/readthedocs-tests/git-repository'
        shutil.rmtree(self.project_repository_path, ignore_errors=True)
        os.makedirs(self.project_repository_path)
        self.patches['models.Project.checkout_path'] = mock.patch('readthedocs.projects.models.Project.checkout_path', return_value=self.project_repository_path)
        self.patches['git.Backend.make_clean_working_dir'] = mock.patch('readthedocs.vcs_support.backends.git.Backend.make_clean_working_dir')
        self.patches['git.Backend.submodules'] = mock.patch('readthedocs.vcs_support.backends.git.Backend.submodules', new_callable=mock.PropertyMock, return_value=['one', 'two', 'three'])

    def _mock_environment(self):
        if False:
            for i in range(10):
                print('nop')
        self.patches['environment.run'] = mock.patch('readthedocs.projects.tasks.builds.LocalBuildEnvironment.run', return_value=mock.MagicMock(successful=True))

    def _mock_storage(self):
        if False:
            return 10
        self.patches['build_media_storage'] = mock.patch('readthedocs.projects.tasks.builds.build_media_storage')

    def _mock_api(self):
        if False:
            while True:
                i = 10
        headers = {'Content-Type': 'application/json'}
        self.requestsmock.get(f'{settings.SLUMBER_API_HOST}/api/v2/version/{self.version.pk}/', json=lambda requests, context: VersionAdminSerializer(self.version).data, headers=headers)
        self.requestsmock.patch(f'{settings.SLUMBER_API_HOST}/api/v2/version/{self.version.pk}/', status_code=201)
        self.requestsmock.get(f'{settings.SLUMBER_API_HOST}/api/v2/build/{self.build.pk}/', json=lambda request, context: {'id': self.build.pk, 'state': BUILD_STATE_TRIGGERED, 'commit': self.build.commit}, headers=headers)
        self.requestsmock.post(f'{settings.SLUMBER_API_HOST}/api/v2/command/', status_code=201)
        self.requestsmock.patch(f'{settings.SLUMBER_API_HOST}/api/v2/build/{self.build.pk}/', status_code=201)
        self.requestsmock.get(f'{settings.SLUMBER_API_HOST}/api/v2/build/concurrent/?project__slug={self.project.slug}', json=lambda request, context: {'limit_reached': False, 'max_concurrent': settings.RTD_MAX_CONCURRENT_BUILDS, 'concurrent': 0}, headers=headers)
        self.requestsmock.get(f'{settings.SLUMBER_API_HOST}/api/v2/project/{self.project.pk}/active_versions/', json=lambda request, context: {'versions': [{'id': self.version.pk, 'slug': self.version.slug}]}, headers=headers)
        self.requestsmock.patch(f'{settings.SLUMBER_API_HOST}/api/v2/project/{self.project.pk}/', status_code=201)
        self.requestsmock.post(f'{settings.SLUMBER_API_HOST}/api/v2/revoke/', status_code=204)