"""
OnionShare | https://onionshare.org/

Copyright (C) 2014-2022 Micah Lee, et al. <micah@micahflee.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from .censorship import CensorshipCircumvention
from .meek import Meek
from stem.control import Controller
from stem import ProtocolError, SocketClosed
from stem.connection import MissingPassword, UnreadableCookieFile, AuthenticationFailure
import base64
import nacl.public
import os
import psutil
import shlex
import subprocess
import tempfile
import time
import traceback
from distutils.version import LooseVersion as Version

class TorErrorAutomatic(Exception):
    """
    OnionShare is failing to connect and authenticate to the Tor controller,
    using automatic settings that should work with Tor Browser.
    """

class TorErrorInvalidSetting(Exception):
    """
    This exception is raised if the settings just don't make sense.
    """

class TorErrorSocketPort(Exception):
    """
    OnionShare can't connect to the Tor controller using the supplied address and port.
    """

class TorErrorSocketFile(Exception):
    """
    OnionShare can't connect to the Tor controller using the supplied socket file.
    """

class TorErrorMissingPassword(Exception):
    """
    OnionShare connected to the Tor controller, but it requires a password.
    """

class TorErrorUnreadableCookieFile(Exception):
    """
    OnionShare connected to the Tor controller, but your user does not have permission
    to access the cookie file.
    """

class TorErrorAuthError(Exception):
    """
    OnionShare connected to the address and port, but can't authenticate. It's possible
    that a Tor controller isn't listening on this port.
    """

class TorErrorProtocolError(Exception):
    """
    This exception is raised if onionshare connects to the Tor controller, but it
    isn't acting like a Tor controller (such as in Whonix).
    """

class TorTooOldEphemeral(Exception):
    """
    This exception is raised if the version of tor doesn't support ephemeral onion services
    """

class TorTooOldStealth(Exception):
    """
    This exception is raised if the version of tor doesn't support stealth onion services
    """

class BundledTorTimeout(Exception):
    """
    This exception is raised if onionshare is set to use the bundled Tor binary,
    but Tor doesn't finish connecting promptly.
    """

class BundledTorCanceled(Exception):
    """
    This exception is raised if onionshare is set to use the bundled Tor binary,
    and the user cancels connecting to Tor
    """

class BundledTorBroken(Exception):
    """
    This exception is raised if onionshare is set to use the bundled Tor binary,
    but the process seems to fail to run.
    """

class PortNotAvailable(Exception):
    """
    There are no available ports for OnionShare to use, which really shouldn't ever happen
    """

class Onion(object):
    """
    Onion is an abstraction layer for connecting to the Tor control port and
    creating onion services. OnionShare supports creating onion services by
    connecting to the Tor controller and using ADD_ONION, DEL_ONION.

    stealth: Should the onion service be stealth?

    settings: A Settings object. If it's not passed in, load from disk.

    bundled_connection_func: If the tor connection type is bundled, optionally
    call this function and pass in a status string while connecting to tor. This
    is necessary for status updates to reach the GUI.
    """

    def __init__(self, common, use_tmp_dir=False, get_tor_paths=None):
        if False:
            while True:
                i = 10
        self.common = common
        self.common.log('Onion', '__init__')
        self.use_tmp_dir = use_tmp_dir
        if not get_tor_paths:
            get_tor_paths = self.common.get_tor_paths
        (self.tor_path, self.tor_geo_ip_file_path, self.tor_geo_ipv6_file_path, self.obfs4proxy_file_path, self.snowflake_file_path, self.meek_client_file_path) = get_tor_paths()
        self.tor_proc = None
        self.c = None
        self.connected_to_tor = False
        self.auth_string = None
        self.graceful_close_onions = []

    def key_str(self, key):
        if False:
            while True:
                i = 10
        '\n        Returns a base32 decoded string of a key.\n        '
        key_bytes = bytes(key)
        key_b32 = base64.b32encode(key_bytes)
        assert key_b32[-4:] == b'===='
        key_b32 = key_b32[:-4]
        s = key_b32.decode('utf-8')
        return s

    def connect(self, custom_settings=None, config=None, tor_status_update_func=None, connect_timeout=120, local_only=False):
        if False:
            print('Hello World!')
        if local_only:
            self.common.log('Onion', 'connect', '--local-only, so skip trying to connect')
            return
        if custom_settings:
            self.settings = custom_settings
        elif config:
            self.common.load_settings(config)
            self.settings = self.common.settings
        else:
            self.common.load_settings()
            self.settings = self.common.settings
        self.common.log('Onion', 'connect', f"connection_type={self.settings.get('connection_type')}")
        self.c = None
        if self.settings.get('connection_type') == 'bundled':
            if self.use_tmp_dir:
                self.tor_data_directory = tempfile.TemporaryDirectory(dir=self.common.build_tmp_dir())
                self.tor_data_directory_name = self.tor_data_directory.name
            else:
                self.tor_data_directory_name = self.common.build_tor_dir()
            self.common.log('Onion', 'connect', f'tor_data_directory_name={self.tor_data_directory_name}')
            with open(self.common.get_resource_path('torrc_template')) as f:
                torrc_template = f.read()
            self.tor_cookie_auth_file = os.path.join(self.tor_data_directory_name, 'cookie')
            try:
                self.tor_socks_port = self.common.get_available_port(1000, 65535)
            except Exception:
                print('OnionShare port not available')
                raise PortNotAvailable()
            self.tor_torrc = os.path.join(self.tor_data_directory_name, 'torrc')
            for proc in psutil.process_iter(['pid', 'name', 'username']):
                try:
                    cmdline = proc.cmdline()
                    if cmdline[0] == self.tor_path and cmdline[1] == '-f' and (cmdline[2] == self.tor_torrc):
                        self.common.log('Onion', 'connect', 'found a stale tor process, killing it')
                        proc.terminate()
                        proc.wait()
                        break
                except Exception:
                    pass
            if self.common.platform == 'Windows' or self.common.platform == 'Darwin':
                torrc_template += 'ControlPort {{control_port}}\n'
                try:
                    self.tor_control_port = self.common.get_available_port(1000, 65535)
                except Exception:
                    print('OnionShare port not available')
                    raise PortNotAvailable()
                self.tor_control_socket = None
            else:
                torrc_template += 'ControlSocket {{control_socket}}\n'
                self.tor_control_port = None
                self.tor_control_socket = os.path.join(self.tor_data_directory_name, 'control_socket')
            torrc_template = torrc_template.replace('{{data_directory}}', self.tor_data_directory_name)
            torrc_template = torrc_template.replace('{{control_port}}', str(self.tor_control_port))
            torrc_template = torrc_template.replace('{{control_socket}}', str(self.tor_control_socket))
            torrc_template = torrc_template.replace('{{cookie_auth_file}}', self.tor_cookie_auth_file)
            torrc_template = torrc_template.replace('{{geo_ip_file}}', self.tor_geo_ip_file_path)
            torrc_template = torrc_template.replace('{{geo_ipv6_file}}', self.tor_geo_ipv6_file_path)
            torrc_template = torrc_template.replace('{{socks_port}}', str(self.tor_socks_port))
            torrc_template = torrc_template.replace('{{obfs4proxy_path}}', str(self.obfs4proxy_file_path))
            torrc_template = torrc_template.replace('{{snowflake_path}}', str(self.snowflake_file_path))
            with open(self.tor_torrc, 'w') as f:
                self.common.log('Onion', 'connect', 'Writing torrc template file')
                f.write(torrc_template)
                if self.settings.get('bridges_enabled'):
                    f.write('\nUseBridges 1\n')
                    if self.settings.get('bridges_type') == 'built-in':
                        use_torrc_bridge_templates = False
                        builtin_bridge_type = self.settings.get('bridges_builtin_pt')
                        if self.settings.get('bridges_builtin'):
                            try:
                                for line in self.settings.get('bridges_builtin')[builtin_bridge_type]:
                                    if line.strip() != '':
                                        f.write(f'Bridge {line}\n')
                                self.common.log('Onion', 'connect', 'Wrote in the built-in bridges from OnionShare settings')
                            except KeyError:
                                use_torrc_bridge_templates = True
                        else:
                            use_torrc_bridge_templates = True
                        if use_torrc_bridge_templates:
                            if builtin_bridge_type == 'obfs4':
                                with open(self.common.get_resource_path('torrc_template-obfs4')) as o:
                                    f.write(o.read())
                            elif builtin_bridge_type == 'meek-azure':
                                with open(self.common.get_resource_path('torrc_template-meek_lite_azure')) as o:
                                    f.write(o.read())
                            elif builtin_bridge_type == 'snowflake':
                                with open(self.common.get_resource_path('torrc_template-snowflake')) as o:
                                    f.write(o.read())
                            self.common.log('Onion', 'connect', 'Wrote in the built-in bridges from torrc templates')
                    elif self.settings.get('bridges_type') == 'moat':
                        for line in self.settings.get('bridges_moat').split('\n'):
                            if line.strip() != '':
                                f.write(f'Bridge {line}\n')
                    elif self.settings.get('bridges_type') == 'custom':
                        for line in self.settings.get('bridges_custom').split('\n'):
                            if line.strip() != '':
                                f.write(f'Bridge {line}\n')
            self.common.log('Onion', 'connect', f'starting {self.tor_path} subprocess')
            start_ts = time.time()
            if self.common.platform == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                self.tor_proc = subprocess.Popen([self.tor_path, '-f', self.tor_torrc], stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
            else:
                if self.common.is_snapcraft():
                    env = None
                else:
                    env = {'LD_LIBRARY_PATH': os.path.dirname(self.tor_path)}
                self.tor_proc = subprocess.Popen([self.tor_path, '-f', self.tor_torrc], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            self.common.log('Onion', 'connect', f'tor pid: {self.tor_proc.pid}')
            time.sleep(2)
            return_code = self.tor_proc.poll()
            if return_code != None:
                self.common.log('Onion', 'connect', f'tor process has terminated early: {return_code}')
            self.common.log('Onion', 'connect', 'authenticating to tor controller')
            try:
                if self.common.platform == 'Windows' or self.common.platform == 'Darwin':
                    self.c = Controller.from_port(port=self.tor_control_port)
                    self.c.authenticate()
                else:
                    self.c = Controller.from_socket_file(path=self.tor_control_socket)
                    self.c.authenticate()
            except Exception as e:
                print('OnionShare could not connect to Tor:\n{}'.format(e.args[0]))
                print(traceback.format_exc())
                raise BundledTorBroken(e.args[0])
            while True:
                try:
                    res = self.c.get_info('status/bootstrap-phase')
                except SocketClosed:
                    raise BundledTorCanceled()
                res_parts = shlex.split(res)
                progress = res_parts[2].split('=')[1]
                summary = res_parts[4].split('=')[1]
                print(f'\rConnecting to the Tor network: {progress}% - {summary}\x1b[K', end='')
                if callable(tor_status_update_func):
                    if not tor_status_update_func(progress, summary):
                        self.common.log('Onion', 'connect', 'tor_status_update_func returned false, canceling connecting to Tor')
                        print()
                        return False
                if summary == 'Done':
                    print('')
                    break
                time.sleep(0.2)
                if self.settings.get('bridges_enabled'):
                    if connect_timeout == 120:
                        connect_timeout = 150
                if time.time() - start_ts > connect_timeout:
                    print('')
                    try:
                        self.tor_proc.terminate()
                        print("Taking too long to connect to Tor. Maybe you aren't connected to the Internet, or have an inaccurate system clock?")
                        raise BundledTorTimeout()
                    except FileNotFoundError:
                        pass
        elif self.settings.get('connection_type') == 'automatic':
            automatic_error = 'Could not connect to the Tor controller. Is Tor Browser (available from torproject.org) running in the background?'
            found_tor = False
            env_port = os.environ.get('TOR_CONTROL_PORT')
            if env_port:
                try:
                    self.c = Controller.from_port(port=int(env_port))
                    found_tor = True
                except Exception:
                    pass
            else:
                try:
                    ports = [9151, 9153, 9051]
                    for port in ports:
                        self.c = Controller.from_port(port=port)
                        found_tor = True
                except Exception:
                    pass
                socket_file_path = ''
                if not found_tor:
                    try:
                        if self.common.platform == 'Darwin':
                            socket_file_path = os.path.expanduser('~/Library/Application Support/TorBrowser-Data/Tor/control.socket')
                        self.c = Controller.from_socket_file(path=socket_file_path)
                        found_tor = True
                    except Exception:
                        pass
            if not found_tor:
                try:
                    if self.common.platform == 'Linux' or self.common.platform == 'BSD':
                        socket_file_path = f'/run/user/{os.geteuid()}/Tor/control.socket'
                    elif self.common.platform == 'Darwin':
                        socket_file_path = f'/run/user/{os.geteuid()}/Tor/control.socket'
                    elif self.common.platform == 'Windows':
                        print(automatic_error)
                        raise TorErrorAutomatic()
                    self.c = Controller.from_socket_file(path=socket_file_path)
                except Exception:
                    print(automatic_error)
                    raise TorErrorAutomatic()
            try:
                self.c.authenticate()
            except Exception:
                print(automatic_error)
                raise TorErrorAutomatic()
        else:
            invalid_settings_error = "Can't connect to Tor controller because your settings don't make sense."
            try:
                if self.settings.get('connection_type') == 'control_port':
                    self.c = Controller.from_port(address=self.settings.get('control_port_address'), port=self.settings.get('control_port_port'))
                elif self.settings.get('connection_type') == 'socket_file':
                    self.c = Controller.from_socket_file(path=self.settings.get('socket_file_path'))
                else:
                    print(invalid_settings_error)
                    raise TorErrorInvalidSetting()
            except Exception:
                if self.settings.get('connection_type') == 'control_port':
                    print("Can't connect to the Tor controller at {}:{}.".format(self.settings.get('control_port_address'), self.settings.get('control_port_port')))
                    raise TorErrorSocketPort(self.settings.get('control_port_address'), self.settings.get('control_port_port'))
                print("Can't connect to the Tor controller using socket file {}.".format(self.settings.get('socket_file_path')))
                raise TorErrorSocketFile(self.settings.get('socket_file_path'))
            try:
                if self.settings.get('auth_type') == 'no_auth':
                    self.c.authenticate()
                elif self.settings.get('auth_type') == 'password':
                    self.c.authenticate(self.settings.get('auth_password'))
                else:
                    print(invalid_settings_error)
                    raise TorErrorInvalidSetting()
            except MissingPassword:
                print('Connected to Tor controller, but it requires a password to authenticate.')
                raise TorErrorMissingPassword()
            except UnreadableCookieFile:
                print('Connected to the Tor controller, but password may be wrong, or your user is not permitted to read the cookie file.')
                raise TorErrorUnreadableCookieFile()
            except AuthenticationFailure:
                print("Connected to {}:{}, but can't authenticate. Maybe this isn't a Tor controller?".format(self.settings.get('control_port_address'), self.settings.get('control_port_port')))
                raise TorErrorAuthError(self.settings.get('control_port_address'), self.settings.get('control_port_port'))
        self.connected_to_tor = True
        self.tor_version = self.c.get_version().version_str
        self.common.log('Onion', 'connect', f'Connected to tor {self.tor_version}')
        list_ephemeral_hidden_services = getattr(self.c, 'list_ephemeral_hidden_services', None)
        self.supports_ephemeral = callable(list_ephemeral_hidden_services) and self.tor_version >= '0.2.7.1'
        try:
            res = self.c.create_ephemeral_hidden_service({1: 1}, basic_auth=None, await_publication=False, key_type='NEW', key_content='ED25519-V3', client_auth_v3='E2GOT5LTUTP3OAMRCRXO4GSH6VKJEUOXZQUC336SRKAHTTT5OVSA')
            tmp_service_id = res.service_id
            self.c.remove_ephemeral_hidden_service(tmp_service_id)
            self.supports_stealth = True
        except Exception:
            self.supports_stealth = False
        self.supports_v3_onions = self.tor_version >= Version('0.3.5.7')
        if self.settings.get('bridges_enabled') and self.settings.get('bridges_type') == 'built-in':
            self.update_builtin_bridges()

    def is_authenticated(self):
        if False:
            print('Hello World!')
        '\n        Returns True if the Tor connection is still working, or False otherwise.\n        '
        if self.c is not None:
            return self.c.is_authenticated()
        else:
            return False

    def start_onion_service(self, mode, mode_settings, port, await_publication):
        if False:
            while True:
                i = 10
        '\n        Start a onion service on port 80, pointing to the given port, and\n        return the onion hostname.\n        '
        self.common.log('Onion', 'start_onion_service', f'port={port}')
        if not self.supports_ephemeral:
            print('Your version of Tor is too old, ephemeral onion services are not supported')
            raise TorTooOldEphemeral()
        if mode_settings.get('onion', 'private_key'):
            key_content = mode_settings.get('onion', 'private_key')
            key_type = 'ED25519-V3'
        else:
            key_content = 'ED25519-V3'
            key_type = 'NEW'
        debug_message = f'key_type={key_type}'
        if key_type == 'NEW':
            debug_message += f', key_content={key_content}'
        self.common.log('Onion', 'start_onion_service', debug_message)
        if mode_settings.get('general', 'public'):
            client_auth_priv_key = None
            client_auth_pub_key = None
        elif not self.supports_stealth:
            print('Your version of Tor is too old, stealth onion services are not supported')
            raise TorTooOldStealth()
        elif key_type == 'NEW' or not mode_settings.get('onion', 'client_auth_priv_key'):
            client_auth_priv_key_raw = nacl.public.PrivateKey.generate()
            client_auth_priv_key = self.key_str(client_auth_priv_key_raw)
            client_auth_pub_key = self.key_str(client_auth_priv_key_raw.public_key)
        else:
            client_auth_priv_key = mode_settings.get('onion', 'client_auth_priv_key')
            client_auth_pub_key = mode_settings.get('onion', 'client_auth_pub_key')
        try:
            if not self.supports_stealth:
                res = self.c.create_ephemeral_hidden_service({80: port}, await_publication=await_publication, basic_auth=None, key_type=key_type, key_content=key_content)
            else:
                res = self.c.create_ephemeral_hidden_service({80: port}, await_publication=await_publication, basic_auth=None, key_type=key_type, key_content=key_content, client_auth_v3=client_auth_pub_key)
        except ProtocolError as e:
            print('Tor error: {}'.format(e.args[0]))
            raise TorErrorProtocolError(e.args[0])
        onion_host = res.service_id + '.onion'
        if mode == 'share':
            self.graceful_close_onions.append(res.service_id)
        mode_settings.set('general', 'service_id', res.service_id)
        if not mode_settings.get('onion', 'private_key'):
            mode_settings.set('onion', 'private_key', res.private_key)
        if not mode_settings.get('general', 'public'):
            mode_settings.set('onion', 'client_auth_priv_key', client_auth_priv_key)
            mode_settings.set('onion', 'client_auth_pub_key', client_auth_pub_key)
            self.auth_string = client_auth_priv_key
        return onion_host

    def stop_onion_service(self, mode_settings):
        if False:
            i = 10
            return i + 15
        '\n        Stop a specific onion service\n        '
        onion_host = mode_settings.get('general', 'service_id')
        if onion_host:
            self.common.log('Onion', 'stop_onion_service', f'onion host: {onion_host}')
            try:
                self.c.remove_ephemeral_hidden_service(mode_settings.get('general', 'service_id'))
            except Exception:
                self.common.log('Onion', 'stop_onion_service', f'failed to remove {onion_host}')

    def cleanup(self, stop_tor=True, wait=True):
        if False:
            i = 10
            return i + 15
        "\n        Stop onion services that were created earlier. If there's a tor subprocess running, kill it.\n        "
        self.common.log('Onion', 'cleanup')
        try:
            onions = self.c.list_ephemeral_hidden_services()
            for service_id in onions:
                onion_host = f'{service_id}.onion'
                try:
                    self.common.log('Onion', 'cleanup', f'trying to remove onion {onion_host}')
                    self.c.remove_ephemeral_hidden_service(service_id)
                except Exception:
                    self.common.log('Onion', 'cleanup', f'failed to remove onion {onion_host}')
                    pass
        except Exception:
            pass
        if stop_tor:
            if self.tor_proc:
                if wait:
                    try:
                        rendezvous_circuit_ids = []
                        for c in self.c.get_circuits():
                            if c.purpose == 'HS_SERVICE_REND' and c.rend_query in self.graceful_close_onions:
                                rendezvous_circuit_ids.append(c.id)
                        symbols = list('\\|/-')
                        symbols_i = 0
                        while True:
                            num_rend_circuits = 0
                            for c in self.c.get_circuits():
                                if c.id in rendezvous_circuit_ids:
                                    num_rend_circuits += 1
                            if num_rend_circuits == 0:
                                print('\rTor rendezvous circuits have closed' + ' ' * 20)
                                break
                            if num_rend_circuits == 1:
                                circuits = 'circuit'
                            else:
                                circuits = 'circuits'
                            print(f'\rWaiting for {num_rend_circuits} Tor rendezvous {circuits} to close {symbols[symbols_i]} ', end='')
                            symbols_i = (symbols_i + 1) % len(symbols)
                            time.sleep(1)
                    except Exception:
                        pass
                self.tor_proc.terminate()
                time.sleep(0.2)
                if self.tor_proc.poll() is None:
                    self.common.log('Onion', 'cleanup', "Tried to terminate tor process but it's still running")
                    try:
                        self.tor_proc.kill()
                        time.sleep(0.2)
                        if self.tor_proc.poll() is None:
                            self.common.log('Onion', 'cleanup', "Tried to kill tor process but it's still running")
                    except Exception:
                        self.common.log('Onion', 'cleanup', 'Exception while killing tor process')
                self.tor_proc = None
            self.connected_to_tor = False
            try:
                if self.use_tmp_dir:
                    self.tor_data_directory.cleanup()
            except Exception:
                pass

    def get_tor_socks_port(self):
        if False:
            while True:
                i = 10
        '\n        Returns a (address, port) tuple for the Tor SOCKS port\n        '
        self.common.log('Onion', 'get_tor_socks_port')
        if self.settings.get('connection_type') == 'bundled':
            return ('127.0.0.1', self.tor_socks_port)
        elif self.settings.get('connection_type') == 'automatic':
            return ('127.0.0.1', 9150)
        else:
            return (self.settings.get('socks_address'), self.settings.get('socks_port'))

    def update_builtin_bridges(self):
        if False:
            i = 10
            return i + 15
        '\n        Use the CensorshipCircumvention API to fetch the latest built-in bridges\n        and update them in settings.\n        '
        builtin_bridges = False
        meek = None
        if self.is_authenticated:
            self.common.log('Onion', 'update_builtin_bridges', 'Updating the built-in bridges. Trying over Tor first')
            self.censorship_circumvention = CensorshipCircumvention(self.common, None, self)
            builtin_bridges = self.censorship_circumvention.request_builtin_bridges()
        if not builtin_bridges:
            self.common.log('Onion', 'update_builtin_bridges', 'Updating the built-in bridges. Trying via Meek (no Tor)')
            meek = Meek(self.common)
            meek.start()
            self.censorship_circumvention = CensorshipCircumvention(self.common, meek, None)
            builtin_bridges = self.censorship_circumvention.request_builtin_bridges()
            meek.cleanup()
        if builtin_bridges:
            self.common.log('Onion', 'update_builtin_bridges', f'Obtained bridges: {builtin_bridges}')
            self.settings.set('bridges_builtin', builtin_bridges)
            self.settings.save()
        else:
            self.common.log('Onion', 'update_builtin_bridges', 'Error getting built-in bridges')
            return False