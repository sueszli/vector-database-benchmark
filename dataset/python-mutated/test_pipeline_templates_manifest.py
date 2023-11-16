from unittest import TestCase
import os
from pathlib import Path
from samcli.commands.pipeline.init.pipeline_templates_manifest import Provider, PipelineTemplatesManifest, PipelineTemplateMetadata, AppPipelineTemplateManifestException
from samcli.lib.utils import osutils
INVALID_YAML_MANIFEST = '\nproviders:\n- Jenkins with wrong identation\n'
MISSING_KEYS_MANIFEST = '\nNotProviders:\n  - Jenkins\nTemplates:\n  - NotName: jenkins-two-environments-pipeline\n    provider: Jenkins\n    location: templates/cookiecutter-jenkins-two-environments-pipeline\n'
VALID_MANIFEST = '\nproviders:\n  - displayName: Jenkins\n    id: jenkins\n  - displayName: Gitlab CI/CD\n    id: gitlab\n  - displayName: Github Actions\n    id: github-actions\ntemplates:\n  - displayName: jenkins-two-environments-pipeline\n    provider: jenkins\n    location: templates/cookiecutter-jenkins-two-environments-pipeline\n  - displayName: gitlab-two-environments-pipeline\n    provider: gitlab\n    location: templates/cookiecutter-gitlab-two-environments-pipeline\n  - displayName: Github-Actions-two-environments-pipeline\n    provider: github-actions\n    location: templates/cookiecutter-github-actions-two-environments-pipeline\n'

class TestCli(TestCase):

    def test_manifest_file_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        non_existing_path = Path(os.path.normpath('/any/non/existing/manifest.yaml'))
        with self.assertRaises(AppPipelineTemplateManifestException):
            PipelineTemplatesManifest(manifest_path=non_existing_path)

    def test_invalid_yaml_manifest_file(self):
        if False:
            for i in range(10):
                print('nop')
        with osutils.mkdir_temp(ignore_errors=True) as tempdir:
            manifest_path = os.path.normpath(os.path.join(tempdir, 'manifest.yaml'))
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                fp.write(INVALID_YAML_MANIFEST)
            with self.assertRaises(AppPipelineTemplateManifestException):
                PipelineTemplatesManifest(manifest_path=Path(manifest_path))

    def test_manifest_missing_required_keys(self):
        if False:
            for i in range(10):
                print('nop')
        with osutils.mkdir_temp(ignore_errors=True) as tempdir:
            manifest_path = os.path.normpath(os.path.join(tempdir, 'manifest.yaml'))
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                fp.write(MISSING_KEYS_MANIFEST)
            with self.assertRaises(AppPipelineTemplateManifestException):
                PipelineTemplatesManifest(manifest_path=Path(manifest_path))

    def test_manifest_happy_case(self):
        if False:
            for i in range(10):
                print('nop')
        with osutils.mkdir_temp(ignore_errors=True) as tempdir:
            manifest_path = os.path.normpath(os.path.join(tempdir, 'manifest.yaml'))
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                fp.write(VALID_MANIFEST)
            manifest = PipelineTemplatesManifest(manifest_path=Path(manifest_path))
        self.assertEqual(len(manifest.providers), 3)
        gitlab_provider: Provider = next((p for p in manifest.providers if p.id == 'gitlab'))
        self.assertEqual(gitlab_provider.display_name, 'Gitlab CI/CD')
        self.assertEqual(len(manifest.templates), 3)
        gitlab_template: PipelineTemplateMetadata = next((t for t in manifest.templates if t.provider == 'gitlab'))
        self.assertEqual(gitlab_template.display_name, 'gitlab-two-environments-pipeline')
        self.assertEqual(gitlab_template.provider, 'gitlab')
        self.assertEqual(gitlab_template.location, 'templates/cookiecutter-gitlab-two-environments-pipeline')