from __future__ import absolute_import
import os
import re
import six
import jsonschema
from st2common import log as logging
from st2common.constants.meta import ALLOWED_EXTS
from st2common.bootstrap.base import ResourceRegistrar
from st2common.persistence.action import Action
from st2common.models.api.action import ActionAPI
from st2common.models.system.common import ResourceReference
import st2common.content.utils as content_utils
import st2common.util.action_db as action_utils
import st2common.validators.api.action as action_validator
__all__ = ['ActionsRegistrar', 'register_actions']
LOG = logging.getLogger(__name__)

class ActionsRegistrar(ResourceRegistrar):
    ALLOWED_EXTENSIONS = ALLOWED_EXTS

    def register_from_packs(self, base_dirs):
        if False:
            print('Hello World!')
        '\n        Discover all the packs in the provided directory and register actions from all of the\n        discovered packs.\n\n        :return: Number of actions registered, Number of actions overridden\n        :rtype: ``tuple``\n        '
        self.register_packs(base_dirs=base_dirs)
        registered_count = 0
        overridden_count = 0
        content = self._pack_loader.get_content(base_dirs=base_dirs, content_type='actions')
        for (pack, actions_dir) in six.iteritems(content):
            if not actions_dir:
                LOG.debug('Pack %s does not contain actions.', pack)
                continue
            try:
                LOG.debug('Registering actions from pack %s:, dir: %s', pack, actions_dir)
                actions = self._get_actions_from_pack(actions_dir)
                (count, overridden) = self._register_actions_from_pack(pack=pack, actions=actions)
                registered_count += count
                overridden_count += overridden
            except Exception as e:
                if self._fail_on_failure:
                    raise e
                LOG.exception('Failed registering all actions from pack: %s', actions_dir)
        return (registered_count, overridden_count)

    def register_from_pack(self, pack_dir):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register all the actions from the provided pack.\n\n        :return: Number of actions registered, Number of actions overridden\n        :rtype: ``tuple``\n        '
        pack_dir = pack_dir[:-1] if pack_dir.endswith('/') else pack_dir
        (_, pack) = os.path.split(pack_dir)
        actions_dir = self._pack_loader.get_content_from_pack(pack_dir=pack_dir, content_type='actions')
        self.register_pack(pack_name=pack, pack_dir=pack_dir)
        registered_count = 0
        overridden_count = 0
        if not actions_dir:
            return registered_count
        LOG.debug('Registering actions from pack %s:, dir: %s', pack, actions_dir)
        try:
            actions = self._get_actions_from_pack(actions_dir=actions_dir)
            (registered_count, overridden_count) = self._register_actions_from_pack(pack=pack, actions=actions)
        except Exception as e:
            if self._fail_on_failure:
                raise e
            LOG.exception('Failed registering all actions from pack: %s', actions_dir)
        return (registered_count, overridden_count)

    def _get_actions_from_pack(self, actions_dir):
        if False:
            print('Hello World!')
        actions = self.get_resources_from_pack(resources_dir=actions_dir)
        config_files = ['actions/config' + ext for ext in self.ALLOWED_EXTENSIONS]
        for config_file in config_files:
            actions = [file_path for file_path in actions if config_file not in file_path]
        return actions

    def _register_action(self, pack, action):
        if False:
            while True:
                i = 10
        content = self._meta_loader.load(action)
        pack_field = content.get('pack', None)
        if not pack_field:
            content['pack'] = pack
            pack_field = pack
        if pack_field != pack:
            raise Exception('Model is in pack "%s" but field "pack" is different: %s' % (pack, pack_field))
        metadata_file = content_utils.get_relative_path_to_pack_file(pack_ref=pack, file_path=action, use_pack_cache=True)
        content['metadata_file'] = metadata_file
        altered = self._override_loader.override(pack, 'actions', content)
        action_api = ActionAPI(**content)
        try:
            action_api.validate()
        except jsonschema.ValidationError as e:
            msg = six.text_type(e)
            is_invalid_parameter_name = 'does not match any of the regexes: ' in msg
            if is_invalid_parameter_name:
                match = re.search("'(.+?)' does not match any of the regexes", msg)
                if match:
                    parameter_name = match.groups()[0]
                else:
                    parameter_name = 'unknown'
                new_msg = 'Parameter name "%s" is invalid. Valid characters for parameter name are [a-zA-Z0-0_].' % parameter_name
                new_msg += '\n\n' + msg
                raise jsonschema.ValidationError(new_msg)
            raise e
        if self._use_runners_cache:
            runner_type_db = self._runner_type_db_cache.get(action_api.runner_type, None)
            if not runner_type_db:
                runner_type_db = action_validator.get_runner_model(action_api)
                self._runner_type_db_cache[action_api.runner_type] = runner_type_db
        else:
            runner_type_db = None
        action_validator.validate_action(action_api, runner_type_db=runner_type_db)
        model = ActionAPI.to_model(action_api)
        action_ref = ResourceReference.to_string_reference(pack=pack, name=str(content['name']))
        existing = action_utils.get_action_by_ref(action_ref, only_fields=['id', 'ref', 'pack'])
        if not existing:
            LOG.debug('Action %s not found. Creating new one with: %s', action_ref, content)
        else:
            LOG.debug('Action %s found. Will be updated from: %s to: %s', action_ref, existing, model)
            model.id = existing.id
        try:
            model = Action.add_or_update(model)
            extra = {'action_db': model}
            LOG.audit('Action updated. Action %s from %s.', model, action, extra=extra)
        except Exception:
            LOG.exception('Failed to write action to db %s.', model.name)
            raise
        return altered

    def _register_actions_from_pack(self, pack, actions):
        if False:
            while True:
                i = 10
        registered_count = 0
        overridden_count = 0
        for action in actions:
            try:
                LOG.debug('Loading action from %s.', action)
                altered = self._register_action(pack=pack, action=action)
                if altered:
                    overridden_count += 1
            except Exception as e:
                if self._fail_on_failure:
                    msg = 'Failed to register action "%s" from pack "%s": %s' % (action, pack, six.text_type(e))
                    raise ValueError(msg)
                LOG.exception('Unable to register action: %s', action)
                continue
            else:
                registered_count += 1
        return (registered_count, overridden_count)

def register_actions(packs_base_paths=None, pack_dir=None, use_pack_cache=True, use_runners_cache=False, fail_on_failure=False):
    if False:
        while True:
            i = 10
    if packs_base_paths:
        if not isinstance(packs_base_paths, list):
            raise ValueError(f'The pack base paths has a value that is not a list (was{type(packs_base_paths)}).')
    if not packs_base_paths:
        packs_base_paths = content_utils.get_packs_base_paths()
    registrar = ActionsRegistrar(use_pack_cache=use_pack_cache, use_runners_cache=use_runners_cache, fail_on_failure=fail_on_failure)
    if pack_dir:
        result = registrar.register_from_pack(pack_dir=pack_dir)
    else:
        result = registrar.register_from_packs(base_dirs=packs_base_paths)
    return result