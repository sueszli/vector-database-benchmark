"""
Actions used for the automation rules.

Each function will receive the following args:

- version: The version object where the action will be applied
- match_result: The result from the match option
- action_arg: An additional argument to apply the action
"""
import structlog
from readthedocs.core.utils import trigger_build
from readthedocs.projects.constants import PRIVATE, PUBLIC
log = structlog.get_logger(__name__)

def activate_version(version, match_result, action_arg, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Sets version as active.\n\n    It triggers a build if the version isn't built.\n    "
    version.active = True
    version.save()
    if not version.built:
        trigger_build(project=version.project, version=version)

def set_default_version(version, match_result, action_arg, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Sets version as the project's default version.\n\n    The version is activated first.\n    "
    activate_version(version, match_result, action_arg)
    project = version.project
    project.default_version = version.slug
    project.save()

def hide_version(version, match_result, action_arg, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets version as hidden.\n\n    It also activates the version and triggers a build.\n    '
    version.hidden = True
    version.save()
    if not version.active:
        activate_version(version, match_result, action_arg)

def set_public_privacy_level(version, match_result, action_arg, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Sets the privacy_level of the version to public.'
    version.privacy_level = PUBLIC
    version.save()

def set_private_privacy_level(version, match_result, action_arg, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Sets the privacy_level of the version to private.'
    version.privacy_level = PRIVATE
    version.save()

def delete_version(version, match_result, action_arg, *args, **kwargs):
    if False:
        print('Hello World!')
    "Delete a version if isn't marked as the default version."
    if version.project.default_version == version.slug:
        log.info('Skipping deleting default version.', project_slug=version.project.slug, version_slug=version.slug)
        return
    version.delete()