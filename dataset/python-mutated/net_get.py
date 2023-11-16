from __future__ import annotations
import os
import re
import uuid
import hashlib
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.plugins.action import ActionBase
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.utils.display import Display
display = Display()

class ActionModule(ActionBase):

    def run(self, tmp=None, task_vars=None):
        if False:
            i = 10
            return i + 15
        socket_path = None
        self._get_network_os(task_vars)
        persistent_connection = self._play_context.connection.split('.')[-1]
        result = super(ActionModule, self).run(task_vars=task_vars)
        if persistent_connection != 'network_cli':
            result['failed'] = True
            result['msg'] = 'connection type %s is not valid for net_get module, please use fully qualified name of network_cli connection type' % self._play_context.connection
            return result
        try:
            src = self._task.args['src']
        except KeyError as exc:
            return {'failed': True, 'msg': 'missing required argument: %s' % exc}
        dest = self._task.args.get('dest')
        if dest is None:
            dest = self._get_default_dest(src)
        else:
            dest = self._handle_dest_path(dest)
        proto = self._task.args.get('protocol')
        if proto is None:
            proto = 'scp'
        if socket_path is None:
            socket_path = self._connection.socket_path
        conn = Connection(socket_path)
        sock_timeout = conn.get_option('persistent_command_timeout')
        try:
            changed = self._handle_existing_file(conn, src, dest, proto, sock_timeout)
            if changed is False:
                result['changed'] = changed
                result['destination'] = dest
                return result
        except Exception as exc:
            result['msg'] = 'Warning: %s idempotency check failed. Check dest' % exc
        try:
            conn.get_file(source=src, destination=dest, proto=proto, timeout=sock_timeout)
        except Exception as exc:
            result['failed'] = True
            result['msg'] = 'Exception received: %s' % exc
        result['changed'] = changed
        result['destination'] = dest
        return result

    def _handle_dest_path(self, dest):
        if False:
            for i in range(10):
                print('nop')
        working_path = self._get_working_path()
        if os.path.isabs(dest) or urlsplit('dest').scheme:
            dst = dest
        else:
            dst = self._loader.path_dwim_relative(working_path, '', dest)
        return dst

    def _get_src_filename_from_path(self, src_path):
        if False:
            return 10
        filename_list = re.split('/|:', src_path)
        return filename_list[-1]

    def _get_default_dest(self, src_path):
        if False:
            return 10
        dest_path = self._get_working_path()
        src_fname = self._get_src_filename_from_path(src_path)
        filename = '%s/%s' % (dest_path, src_fname)
        return filename

    def _handle_existing_file(self, conn, source, dest, proto, timeout):
        if False:
            return 10
        '\n        Determines whether the source and destination file match.\n\n        :return: False if source and dest both exist and have matching sha1 sums, True otherwise.\n        '
        if not os.path.exists(dest):
            return True
        cwd = self._loader.get_basedir()
        filename = str(uuid.uuid4())
        tmp_dest_file = os.path.join(cwd, filename)
        try:
            conn.get_file(source=source, destination=tmp_dest_file, proto=proto, timeout=timeout)
        except ConnectionError as exc:
            error = to_text(exc)
            if error.endswith('No such file or directory'):
                if os.path.exists(tmp_dest_file):
                    os.remove(tmp_dest_file)
                return True
        try:
            with open(tmp_dest_file, 'r') as f:
                new_content = f.read()
            with open(dest, 'r') as f:
                old_content = f.read()
        except (IOError, OSError):
            os.remove(tmp_dest_file)
            raise
        sha1 = hashlib.sha1()
        old_content_b = to_bytes(old_content, errors='surrogate_or_strict')
        sha1.update(old_content_b)
        checksum_old = sha1.digest()
        sha1 = hashlib.sha1()
        new_content_b = to_bytes(new_content, errors='surrogate_or_strict')
        sha1.update(new_content_b)
        checksum_new = sha1.digest()
        os.remove(tmp_dest_file)
        if checksum_old == checksum_new:
            return False
        return True

    def _get_working_path(self):
        if False:
            print('Hello World!')
        cwd = self._loader.get_basedir()
        if self._task._role is not None:
            cwd = self._task._role._role_path
        return cwd

    def _get_network_os(self, task_vars):
        if False:
            while True:
                i = 10
        if 'network_os' in self._task.args and self._task.args['network_os']:
            display.vvvv('Getting network OS from task argument')
            network_os = self._task.args['network_os']
        elif self._play_context.network_os:
            display.vvvv('Getting network OS from inventory')
            network_os = self._play_context.network_os
        elif 'network_os' in task_vars.get('ansible_facts', {}) and task_vars['ansible_facts']['network_os']:
            display.vvvv('Getting network OS from fact')
            network_os = task_vars['ansible_facts']['network_os']
        else:
            raise AnsibleError('ansible_network_os must be specified on this host')
        return network_os