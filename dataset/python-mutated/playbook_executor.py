from __future__ import annotations
import os
from ansible import constants as C
from ansible import context
from ansible.executor.task_queue_manager import TaskQueueManager, AnsibleEndPlay
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.loader import become_loader, connection_loader, shell_loader
from ansible.playbook import Playbook
from ansible.template import Templar
from ansible.utils.helpers import pct_to_int
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.utils.path import makedirs_safe
from ansible.utils.ssh_functions import set_default_transport
from ansible.utils.display import Display
display = Display()

class PlaybookExecutor:
    """
    This is the primary class for executing playbooks, and thus the
    basis for bin/ansible-playbook operation.
    """

    def __init__(self, playbooks, inventory, variable_manager, loader, passwords):
        if False:
            i = 10
            return i + 15
        self._playbooks = playbooks
        self._inventory = inventory
        self._variable_manager = variable_manager
        self._loader = loader
        self.passwords = passwords
        self._unreachable_hosts = dict()
        if context.CLIARGS.get('listhosts') or context.CLIARGS.get('listtasks') or context.CLIARGS.get('listtags') or context.CLIARGS.get('syntax'):
            self._tqm = None
        else:
            self._tqm = TaskQueueManager(inventory=inventory, variable_manager=variable_manager, loader=loader, passwords=self.passwords, forks=context.CLIARGS.get('forks'))
        set_default_transport()

    def run(self):
        if False:
            return 10
        '\n        Run the given playbook, based on the settings in the play which\n        may limit the runs to serialized groups, etc.\n        '
        result = 0
        entrylist = []
        entry = {}
        try:
            list(connection_loader.all(class_only=True))
            list(shell_loader.all(class_only=True))
            list(become_loader.all(class_only=True))
            for playbook in self._playbooks:
                resource = _get_collection_playbook_path(playbook)
                if resource is not None:
                    playbook_path = resource[1]
                    playbook_collection = resource[2]
                else:
                    playbook_path = playbook
                    playbook_collection = _get_collection_name_from_path(playbook)
                if playbook_collection:
                    display.v('running playbook inside collection {0}'.format(playbook_collection))
                    AnsibleCollectionConfig.default_collection = playbook_collection
                else:
                    AnsibleCollectionConfig.default_collection = None
                pb = Playbook.load(playbook_path, variable_manager=self._variable_manager, loader=self._loader)
                if self._tqm is None:
                    entry = {'playbook': playbook_path}
                    entry['plays'] = []
                else:
                    self._tqm.load_callbacks()
                    self._tqm.send_callback('v2_playbook_on_start', pb)
                i = 1
                plays = pb.get_plays()
                display.vv(u'%d plays in %s' % (len(plays), to_text(playbook_path)))
                for play in plays:
                    if play._included_path is not None:
                        self._loader.set_basedir(play._included_path)
                    else:
                        self._loader.set_basedir(pb._basedir)
                    self._inventory.remove_restriction()
                    all_vars = self._variable_manager.get_vars(play=play)
                    templar = Templar(loader=self._loader, variables=all_vars)
                    setattr(play, 'vars_prompt', templar.template(play.vars_prompt))
                    if play.vars_prompt:
                        for var in play.vars_prompt:
                            vname = var['name']
                            prompt = var.get('prompt', vname)
                            default = var.get('default', None)
                            private = boolean(var.get('private', True))
                            confirm = boolean(var.get('confirm', False))
                            encrypt = var.get('encrypt', None)
                            salt_size = var.get('salt_size', None)
                            salt = var.get('salt', None)
                            unsafe = boolean(var.get('unsafe', False))
                            if vname not in self._variable_manager.extra_vars:
                                if self._tqm:
                                    self._tqm.send_callback('v2_playbook_on_vars_prompt', vname, private, prompt, encrypt, confirm, salt_size, salt, default, unsafe)
                                    play.vars[vname] = display.do_var_prompt(vname, private, prompt, encrypt, confirm, salt_size, salt, default, unsafe)
                                else:
                                    play.vars[vname] = default
                    all_vars = self._variable_manager.get_vars(play=play)
                    templar = Templar(loader=self._loader, variables=all_vars)
                    play.post_validate(templar)
                    if context.CLIARGS['syntax']:
                        continue
                    if self._tqm is None:
                        entry['plays'].append(play)
                    else:
                        self._tqm._unreachable_hosts.update(self._unreachable_hosts)
                        previously_failed = len(self._tqm._failed_hosts)
                        previously_unreachable = len(self._tqm._unreachable_hosts)
                        break_play = False
                        batches = self._get_serialized_batches(play)
                        if len(batches) == 0:
                            self._tqm.send_callback('v2_playbook_on_play_start', play)
                            self._tqm.send_callback('v2_playbook_on_no_hosts_matched')
                        for batch in batches:
                            self._inventory.restrict_to_hosts(batch)
                            try:
                                result = self._tqm.run(play=play)
                            except AnsibleEndPlay as e:
                                result = e.result
                                break
                            if result & self._tqm.RUN_FAILED_BREAK_PLAY != 0:
                                result = self._tqm.RUN_FAILED_HOSTS
                                break_play = True
                            failed_hosts_count = len(self._tqm._failed_hosts) + len(self._tqm._unreachable_hosts) - (previously_failed + previously_unreachable)
                            if len(batch) == failed_hosts_count:
                                break_play = True
                                break
                            previously_failed += len(self._tqm._failed_hosts) - previously_failed
                            previously_unreachable += len(self._tqm._unreachable_hosts) - previously_unreachable
                            self._unreachable_hosts.update(self._tqm._unreachable_hosts)
                        if break_play:
                            break
                    i = i + 1
                if entry:
                    entrylist.append(entry)
                if self._tqm is not None:
                    if C.RETRY_FILES_ENABLED:
                        retries = set(self._tqm._failed_hosts.keys())
                        retries.update(self._tqm._unreachable_hosts.keys())
                        retries = sorted(retries)
                        if len(retries) > 0:
                            if C.RETRY_FILES_SAVE_PATH:
                                basedir = C.RETRY_FILES_SAVE_PATH
                            elif playbook_path:
                                basedir = os.path.dirname(os.path.abspath(playbook_path))
                            else:
                                basedir = '~/'
                            (retry_name, ext) = os.path.splitext(os.path.basename(playbook_path))
                            filename = os.path.join(basedir, '%s.retry' % retry_name)
                            if self._generate_retry_inventory(filename, retries):
                                display.display('\tto retry, use: --limit @%s\n' % filename)
                    self._tqm.send_callback('v2_playbook_on_stats', self._tqm._stats)
                if result != 0:
                    break
            if entrylist:
                return entrylist
        finally:
            if self._tqm is not None:
                self._tqm.cleanup()
            if self._loader:
                self._loader.cleanup_all_tmp_files()
        if context.CLIARGS['syntax']:
            display.display('No issues encountered')
            return result
        if context.CLIARGS['start_at_task'] and (not self._tqm._start_at_done):
            display.error('No matching task "%s" found. Note: --start-at-task can only follow static includes.' % context.CLIARGS['start_at_task'])
        return result

    def _get_serialized_batches(self, play):
        if False:
            print('Hello World!')
        '\n        Returns a list of hosts, subdivided into batches based on\n        the serial size specified in the play.\n        '
        all_hosts = self._inventory.get_hosts(play.hosts, order=play.order)
        all_hosts_len = len(all_hosts)
        serial_batch_list = play.serial
        if len(serial_batch_list) == 0:
            serial_batch_list = [-1]
        cur_item = 0
        serialized_batches = []
        while len(all_hosts) > 0:
            serial = pct_to_int(serial_batch_list[cur_item], all_hosts_len)
            if serial <= 0:
                serialized_batches.append(all_hosts)
                break
            else:
                play_hosts = []
                for x in range(serial):
                    if len(all_hosts) > 0:
                        play_hosts.append(all_hosts.pop(0))
                serialized_batches.append(play_hosts)
            cur_item += 1
            if cur_item > len(serial_batch_list) - 1:
                cur_item = len(serial_batch_list) - 1
        return serialized_batches

    def _generate_retry_inventory(self, retry_path, replay_hosts):
        if False:
            while True:
                i = 10
        '\n        Called when a playbook run fails. It generates an inventory which allows\n        re-running on ONLY the failed hosts.  This may duplicate some variable\n        information in group_vars/host_vars but that is ok, and expected.\n        '
        try:
            makedirs_safe(os.path.dirname(retry_path))
            with open(retry_path, 'w') as fd:
                for x in replay_hosts:
                    fd.write('%s\n' % x)
        except Exception as e:
            display.warning("Could not create retry file '%s'.\n\t%s" % (retry_path, to_text(e)))
            return False
        return True