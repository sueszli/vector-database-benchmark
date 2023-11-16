from .. import errors
from .. import utils
from ..types import CancellableStream

class ExecApiMixin:

    @utils.check_resource('container')
    def exec_create(self, container, cmd, stdout=True, stderr=True, stdin=False, tty=False, privileged=False, user='', environment=None, workdir=None, detach_keys=None):
        if False:
            return 10
        '\n        Sets up an exec instance in a running container.\n\n        Args:\n            container (str): Target container where exec instance will be\n                created\n            cmd (str or list): Command to be executed\n            stdout (bool): Attach to stdout. Default: ``True``\n            stderr (bool): Attach to stderr. Default: ``True``\n            stdin (bool): Attach to stdin. Default: ``False``\n            tty (bool): Allocate a pseudo-TTY. Default: False\n            privileged (bool): Run as privileged.\n            user (str): User to execute command as. Default: root\n            environment (dict or list): A dictionary or a list of strings in\n                the following format ``["PASSWORD=xxx"]`` or\n                ``{"PASSWORD": "xxx"}``.\n            workdir (str): Path to working directory for this exec session\n            detach_keys (str): Override the key sequence for detaching\n                a container. Format is a single character `[a-Z]`\n                or `ctrl-<value>` where `<value>` is one of:\n                `a-z`, `@`, `^`, `[`, `,` or `_`.\n                ~/.docker/config.json is used by default.\n\n        Returns:\n            (dict): A dictionary with an exec ``Id`` key.\n\n        Raises:\n            :py:class:`docker.errors.APIError`\n                If the server returns an error.\n        '
        if environment is not None and utils.version_lt(self._version, '1.25'):
            raise errors.InvalidVersion('Setting environment for exec is not supported in API < 1.25')
        if isinstance(cmd, str):
            cmd = utils.split_command(cmd)
        if isinstance(environment, dict):
            environment = utils.utils.format_environment(environment)
        data = {'Container': container, 'User': user, 'Privileged': privileged, 'Tty': tty, 'AttachStdin': stdin, 'AttachStdout': stdout, 'AttachStderr': stderr, 'Cmd': cmd, 'Env': environment}
        if workdir is not None:
            if utils.version_lt(self._version, '1.35'):
                raise errors.InvalidVersion('workdir is not supported for API version < 1.35')
            data['WorkingDir'] = workdir
        if detach_keys:
            data['detachKeys'] = detach_keys
        elif 'detachKeys' in self._general_configs:
            data['detachKeys'] = self._general_configs['detachKeys']
        url = self._url('/containers/{0}/exec', container)
        res = self._post_json(url, data=data)
        return self._result(res, True)

    def exec_inspect(self, exec_id):
        if False:
            i = 10
            return i + 15
        '\n        Return low-level information about an exec command.\n\n        Args:\n            exec_id (str): ID of the exec instance\n\n        Returns:\n            (dict): Dictionary of values returned by the endpoint.\n\n        Raises:\n            :py:class:`docker.errors.APIError`\n                If the server returns an error.\n        '
        if isinstance(exec_id, dict):
            exec_id = exec_id.get('Id')
        res = self._get(self._url('/exec/{0}/json', exec_id))
        return self._result(res, True)

    def exec_resize(self, exec_id, height=None, width=None):
        if False:
            print('Hello World!')
        '\n        Resize the tty session used by the specified exec command.\n\n        Args:\n            exec_id (str): ID of the exec instance\n            height (int): Height of tty session\n            width (int): Width of tty session\n        '
        if isinstance(exec_id, dict):
            exec_id = exec_id.get('Id')
        params = {'h': height, 'w': width}
        url = self._url('/exec/{0}/resize', exec_id)
        res = self._post(url, params=params)
        self._raise_for_status(res)

    @utils.check_resource('exec_id')
    def exec_start(self, exec_id, detach=False, tty=False, stream=False, socket=False, demux=False):
        if False:
            while True:
                i = 10
        '\n        Start a previously set up exec instance.\n\n        Args:\n            exec_id (str): ID of the exec instance\n            detach (bool): If true, detach from the exec command.\n                Default: False\n            tty (bool): Allocate a pseudo-TTY. Default: False\n            stream (bool): Return response data progressively as an iterator\n                of strings, rather than a single string.\n            socket (bool): Return the connection socket to allow custom\n                read/write operations. Must be closed by the caller when done.\n            demux (bool): Return stdout and stderr separately\n\n        Returns:\n\n            (generator or str or tuple): If ``stream=True``, a generator\n            yielding response chunks. If ``socket=True``, a socket object for\n            the connection. A string containing response data otherwise. If\n            ``demux=True``, a tuple with two elements of type byte: stdout and\n            stderr.\n\n        Raises:\n            :py:class:`docker.errors.APIError`\n                If the server returns an error.\n        '
        data = {'Tty': tty, 'Detach': detach}
        headers = {} if detach else {'Connection': 'Upgrade', 'Upgrade': 'tcp'}
        res = self._post_json(self._url('/exec/{0}/start', exec_id), headers=headers, data=data, stream=True)
        if detach:
            try:
                return self._result(res)
            finally:
                res.close()
        if socket:
            return self._get_raw_response_socket(res)
        output = self._read_from_socket(res, stream, tty=tty, demux=demux)
        if stream:
            return CancellableStream(output, res)
        else:
            return output