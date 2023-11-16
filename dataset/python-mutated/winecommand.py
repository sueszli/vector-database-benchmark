import os
import shutil
import stat
import subprocess
import tempfile
from typing import Optional
from bottles.backend.globals import Paths, gamemode_available, gamescope_available, mangohud_available, obs_vkc_available, vmtouch_available
from bottles.backend.logger import Logger
from bottles.backend.managers.runtime import RuntimeManager
from bottles.backend.managers.sandbox import SandboxManager
from bottles.backend.models.config import BottleConfig
from bottles.backend.models.result import Result
from bottles.backend.utils.display import DisplayUtils
from bottles.backend.utils.generic import detect_encoding
from bottles.backend.utils.gpu import GPUUtils
from bottles.backend.utils.manager import ManagerUtils
from bottles.backend.utils.terminal import TerminalUtils
from bottles.backend.utils.steam import SteamUtils
logging = Logger()

class WineEnv:
    """
    This class is used to store and return a command environment.
    """
    __env: dict = {}
    __result: dict = {'envs': {}, 'overrides': []}

    def __init__(self, clean: bool=False):
        if False:
            print('Hello World!')
        self.__env = {}
        if not clean:
            self.__env = os.environ.copy()

    def add(self, key, value, override=False):
        if False:
            return 10
        if key in self.__env:
            if override:
                self.__result['overrides'].append(f'{key}={value}')
            else:
                return
        self.__env[key] = value

    def add_bundle(self, bundle, override=False):
        if False:
            while True:
                i = 10
        for (key, value) in bundle.items():
            self.add(key, value, override)

    def get(self):
        if False:
            i = 10
            return i + 15
        result = self.__result
        result['count_envs'] = len(result['envs'])
        result['count_overrides'] = len(result['overrides'])
        result['envs'] = self.__env
        return result

    def remove(self, key):
        if False:
            return 10
        if key in self.__env:
            del self.__env[key]

    def is_empty(self, key):
        if False:
            print('Hello World!')
        return len(self.__env.get(key, '').strip()) == 0

    def concat(self, key, values, sep=':'):
        if False:
            while True:
                i = 10
        if isinstance(values, str):
            values = [values]
        values = sep.join(values)
        if self.has(key):
            values = self.__env[key] + sep + values
        self.add(key, values, True)

    def has(self, key):
        if False:
            for i in range(10):
                print('nop')
        return key in self.__env

class WineCommand:
    """
    This class is used to run a wine command with a custom environment.
    It also handles the launch in a terminal or not.
    """

    def __init__(self, config: BottleConfig, command: str, terminal: bool=False, arguments: str=False, environment: dict=False, communicate: bool=False, cwd: Optional[str]=None, colors: str='default', minimal: bool=False, post_script: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self.config = self._get_config(config)
        self.minimal = minimal
        self.arguments = arguments
        self.cwd = self._get_cwd(cwd)
        (self.runner, self.runner_runtime) = self._get_runner_info()
        self.command = self.get_cmd(command, post_script)
        self.terminal = terminal
        self.env = self.get_env(environment)
        self.communicate = communicate
        self.colors = colors
        self.vmtouch_files = None

    def _get_config(self, config: BottleConfig) -> BottleConfig:
        if False:
            print('Hello World!')
        if (cnf := config.data.get('config')):
            return cnf
        if not isinstance(config, BottleConfig):
            logging.error('Invalid config type: %s' % type(config))
            return BottleConfig()
        return config

    def _get_cwd(self, cwd) -> str:
        if False:
            return 10
        config = self.config
        if config.Environment == 'Steam':
            bottle = config.Path
        else:
            bottle = ManagerUtils.get_bottle_path(config)
        if not cwd:
            '\n            If no cwd is given, use the WorkingDir from the\n            bottle configuration.\n            '
            cwd = config.WorkingDir
        if cwd == '' or not os.path.exists(cwd):
            '\n            If the WorkingDir is empty, use the bottle path as\n            working directory.\n            '
            cwd = bottle
        return cwd

    def get_env(self, environment: Optional[dict]=None, return_steam_env: bool=False, return_clean_env: bool=False) -> dict:
        if False:
            i = 10
            return i + 15
        env = WineEnv(clean=return_steam_env or return_clean_env)
        config = self.config
        arch = config.Arch
        params = config.Parameters
        if None in [arch, params]:
            return env.get()['envs']
        if environment is None:
            environment = {}
        bottle = ManagerUtils.get_bottle_path(config)
        runner_path = ManagerUtils.get_runner_path(config.Runner)
        if config.Environment == 'Steam':
            bottle = config.Path
            runner_path = config.RunnerPath
        if SteamUtils.is_proton(runner_path):
            runner_path = SteamUtils.get_dist_directory(runner_path)
        dll_overrides = []
        gpu = GPUUtils().get_gpu()
        is_nvidia = DisplayUtils.check_nvidia_device()
        ld = []
        if config.Environment_Variables:
            for (key, value) in config.Environment_Variables.items():
                env.add(key, value, override=True)
                if key == 'WINEDLLOVERRIDES' and value:
                    dll_overrides.extend(value.split(';'))
        if environment:
            if environment.get('WINEDLLOVERRIDES'):
                dll_overrides.append(environment['WINEDLLOVERRIDES'])
                del environment['WINEDLLOVERRIDES']
            if environment.get('DXVK_CONFIG_FILE', '') == 'bottle_root':
                environment['DXVK_CONFIG_FILE'] = os.path.join(bottle, 'dxvk.conf')
            for e in environment:
                env.add(e, environment[e], override=True)
        if config.Language != 'sys':
            env.add('LC_ALL', config.Language)
        if config.DLL_Overrides:
            for (k, v) in config.DLL_Overrides.items():
                dll_overrides.append(f'{k}={v}')
        if not return_steam_env:
            dll_overrides.append("winemenubuilder=''")
        if (params.use_runtime or params.use_eac_runtime or params.use_be_runtime) and (not self.terminal) and (not return_steam_env):
            _rb = RuntimeManager.get_runtime_env('bottles')
            if _rb:
                _eac = RuntimeManager.get_eac()
                _be = RuntimeManager.get_be()
                if params.use_runtime:
                    logging.info('Using Bottles runtime')
                    ld += _rb
                if _eac and (not self.minimal):
                    logging.info('Using EasyAntiCheat runtime')
                    env.add('PROTON_EAC_RUNTIME', _eac)
                    dll_overrides.append('easyanticheat_x86,easyanticheat_x64=b,n')
                if _be and (not self.minimal):
                    logging.info('Using BattlEye runtime')
                    env.add('PROTON_BATTLEYE_RUNTIME', _be)
                    dll_overrides.append('beclient,beclient_x64=b,n')
            else:
                logging.warning('Bottles runtime was requested but not found')
        if arch == 'win64':
            runner_libs = ['lib', 'lib64', 'lib/wine/x86_64-unix', 'lib32/wine/x86_64-unix', 'lib64/wine/x86_64-unix', 'lib/wine/i386-unix', 'lib32/wine/i386-unix', 'lib64/wine/i386-unix']
            gst_libs = ['lib64/gstreamer-1.0', 'lib/gstreamer-1.0', 'lib32/gstreamer-1.0']
        else:
            runner_libs = ['lib', 'lib/wine/i386-unix', 'lib32/wine/i386-unix', 'lib64/wine/i386-unix']
            gst_libs = ['lib/gstreamer-1.0', 'lib32/gstreamer-1.0']
        for lib in runner_libs:
            _path = os.path.join(runner_path, lib)
            if os.path.exists(_path):
                ld.append(_path)
        if not env.has('BOTTLES_USE_SYSTEM_GSTREAMER') and (not return_steam_env):
            gst_env_path = []
            for lib in gst_libs:
                if os.path.exists(os.path.join(runner_path, lib)):
                    gst_env_path.append(os.path.join(runner_path, lib))
            if len(gst_env_path) > 0:
                env.add('GST_PLUGIN_SYSTEM_PATH', ':'.join(gst_env_path), override=True)
        if params.dxvk and (not return_steam_env):
            env.add('WINE_LARGE_ADDRESS_AWARE', '1')
            env.add('DXVK_STATE_CACHE_PATH', os.path.join(bottle, 'cache', 'dxvk_state'))
            env.add('STAGING_SHARED_MEMORY', '1')
            env.add('__GL_SHADER_DISK_CACHE', '1')
            env.add('__GL_SHADER_DISK_CACHE_SKIP_CLEANUP', '1')
            env.add('__GL_SHADER_DISK_CACHE_PATH', os.path.join(bottle, 'cache', 'gl_shader'))
            env.add('MESA_SHADER_CACHE_DIR', os.path.join(bottle, 'cache', 'mesa_shader'))
        if params.vkd3d and (not return_steam_env):
            env.add('VKD3D_SHADER_CACHE_PATH', os.path.join(bottle, 'cache', 'vkd3d_shader'))
        if params.latencyflex and (not return_steam_env):
            _lf_path = ManagerUtils.get_latencyflex_path(config.LatencyFleX)
            _lf_layer_path = os.path.join(_lf_path, 'layer/usr/share/vulkan/implicit_layer.d')
            env.concat('VK_ADD_LAYER_PATH', _lf_layer_path)
            env.add('LFX', '1')
            ld.append(os.path.join(_lf_path, 'layer/usr/lib/x86_64-linux-gnu'))
        else:
            env.add('DISABLE_LFX', '1')
        if params.mangohud and (not self.minimal) and (not (gamescope_available and params.gamescope)):
            env.add('MANGOHUD', '1')
            env.add('MANGOHUD_DLSYM', '1')
        if params.vkbasalt and (not self.minimal):
            vkbasalt_conf_path = os.path.join(ManagerUtils.get_bottle_path(config), 'vkBasalt.conf')
            if os.path.isfile(vkbasalt_conf_path):
                env.add('VKBASALT_CONFIG_FILE', vkbasalt_conf_path)
            env.add('ENABLE_VKBASALT', '1')
        if params.obsvkc and (not self.minimal):
            env.add('OBS_VKCAPTURE', '1')
            if DisplayUtils.display_server_type() == 'x11':
                env.add('OBS_USE_EGL', '1')
        if params.dxvk_nvapi and (not return_steam_env):
            conf = self.__set_dxvk_nvapi_conf(bottle)
            env.add('DXVK_CONFIG_FILE', conf)
            env.add('DXVK_NVAPIHACK', '0')
            env.add('DXVK_ENABLE_NVAPI', '1')
        if params.sync == 'esync':
            env.add('WINEESYNC', '1')
        if params.sync == 'fsync':
            env.add('WINEFSYNC', '1')
        if not return_steam_env:
            debug_level = 'fixme-all'
            if params.fixme_logs:
                debug_level = '+fixme-all'
            env.add('WINEDEBUG', debug_level)
        if params.fsr:
            env.add('WINE_FULLSCREEN_FSR', '1')
            env.add('WINE_FULLSCREEN_FSR_STRENGTH', str(params.fsr_sharpening_strength))
            if params.fsr_quality_mode:
                env.add('WINE_FULLSCREEN_FSR_MODE', str(params.fsr_quality_mode))
        if params.pulseaudio_latency:
            env.add('PULSE_LATENCY_MSEC', '60')
        if not return_steam_env:
            if params.discrete_gpu:
                discrete = gpu['prime']['discrete']
                if discrete is not None:
                    gpu_envs = discrete['envs']
                    for p in gpu_envs:
                        env.add(p, gpu_envs[p])
                    env.concat('VK_ICD_FILENAMES', discrete['icd'])
            if not env.has('VK_ICD_FILENAMES'):
                if gpu['prime']['integrated'] is not None:
                    '\n                    System support PRIME but user disabled the discrete GPU\n                    setting (previus check skipped), so using the integrated one.\n                    '
                    env.concat('VK_ICD_FILENAMES', gpu['prime']['integrated']['icd'])
                else:
                    "\n                    System doesn't support PRIME, so using the first result\n                    from the gpu vendors list.\n                    "
                    if 'vendors' in gpu and len(gpu['vendors']) > 0:
                        _first = list(gpu['vendors'].keys())[0]
                        env.concat('VK_ICD_FILENAMES', gpu['vendors'][_first]['icd'])
                    else:
                        logging.warning('No GPU vendor found, keep going without setting VK_ICD_FILENAMES…')
            if ld:
                env.concat('LD_LIBRARY_PATH', ld)
        env.concat('WINEDLLOVERRIDES', dll_overrides, sep=';')
        if env.is_empty('WINEDLLOVERRIDES'):
            env.remove('WINEDLLOVERRIDES')
        if not return_steam_env:
            env.add('WINEPREFIX', bottle, override=True)
            env.add('WINEARCH', arch)
        return env.get()['envs']

    def _get_runner_info(self) -> tuple[str, str]:
        if False:
            print('Hello World!')
        config = self.config
        runner = ManagerUtils.get_runner_path(config.Runner)
        arch = config.Arch
        runner_runtime = ''
        if config.Environment == 'Steam':
            runner = config.RunnerPath
        if runner in [None, '']:
            return ('', '')
        if SteamUtils.is_proton(runner):
            '\n            If the runner is Proton, set the path to /dist or /files \n            based on check if files exists.\n            Additionally, check for its corresponding runtime.\n            '
            runner_runtime = SteamUtils.get_associated_runtime(runner)
            runner = os.path.join(SteamUtils.get_dist_directory(runner), f'bin/wine')
        elif runner.startswith('sys-'):
            '\n            If the runner type is system, set the runner binary\n            path to the system command. Else set it to the full path.\n            '
            runner = shutil.which('wine')
        else:
            runner = f'{runner}/bin/wine'
        if arch == 'win64':
            runner = f'{runner}64'
        runner = runner.replace(' ', '\\ ')
        return (runner, runner_runtime)

    def get_cmd(self, command, post_script: Optional[str]=None, return_steam_cmd: bool=False, return_clean_cmd: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        config = self.config
        params = config.Parameters
        runner = self.runner
        if return_clean_cmd:
            return_steam_cmd = True
        if not return_steam_cmd and (not return_clean_cmd):
            command = f'{runner} {command}'
        if not self.minimal:
            if gamemode_available and params.gamemode:
                if not return_steam_cmd:
                    command = f'{gamemode_available} {command}'
                else:
                    command = f'gamemode {command}'
            if mangohud_available and params.mangohud and (not params.gamescope):
                if not return_steam_cmd:
                    command = f'{mangohud_available} {command}'
                else:
                    command = f'mangohud {command}'
            if gamescope_available and params.gamescope:
                gamescope_run = tempfile.NamedTemporaryFile(mode='w', suffix='.sh').name
                file = [f'#!/usr/bin/env sh\n']
                if mangohud_available and params.mangohud:
                    file.append(f'{command}&\nmangoapp')
                else:
                    file.append(command)
                with open(gamescope_run, 'w') as f:
                    f.write(''.join(file))
                command = f'{self._get_gamescope_cmd(return_steam_cmd)} -- {gamescope_run}'
                logging.info(f"Running Gamescope command: '{command}'")
                logging.info(f'{gamescope_run} contains:')
                with open(gamescope_run, 'r') as f:
                    logging.info(f'\n\n{f.read()}')
                st = os.stat(gamescope_run)
                os.chmod(gamescope_run, st.st_mode | stat.S_IEXEC)
            if obs_vkc_available and params.obsvkc:
                command = f'{obs_vkc_available} {command}'
        if params.use_steam_runtime:
            _rs = RuntimeManager.get_runtimes('steam')
            _picked = {}
            if _rs:
                if 'sniper' in _rs.keys() and 'sniper' in self.runner_runtime:
                    '\n                    Sniper is the default runtime used by Proton version >= 8.0\n                    '
                    _picked = _rs['sniper']
                elif 'soldier' in _rs.keys() and 'soldier' in self.runner_runtime:
                    '\n                    Sniper is the default runtime used by Proton version >= 5.13 and < 8.0\n                    '
                    _picked = _rs['soldier']
                elif 'scout' in _rs.keys():
                    '\n                    For Wine runners, we cannot make assumption about which runtime would suits\n                    them the best, as it would depend on their build environment.\n                    Sniper/Soldier are not backward-compatible, defaulting to Scout should maximize compatibility.\n                    '
                    _picked = _rs['scout']
            else:
                logging.warning('Steam runtime was requested but not found')
            if _picked:
                logging.info(f"Using Steam runtime {_picked['name']}")
                command = f"{_picked['entry_point']} {command}"
            else:
                logging.warning('Steam runtime was requested and found but there are no valid combinations')
        if self.arguments:
            if '%command%' in self.arguments:
                prefix = self.arguments.split('%command%')[0]
                suffix = self.arguments.split('%command%')[1]
                command = f'{prefix} {command} {suffix}'
            else:
                command = f'{command} {self.arguments}'
        if post_script is not None:
            command = f"{command} ; sh '{post_script}'"
        return command

    def _get_gamescope_cmd(self, return_steam_cmd: bool=False) -> str:
        if False:
            return 10
        config = self.config
        params = config.Parameters
        gamescope_cmd = []
        if gamescope_available and params.gamescope:
            gamescope_cmd = [gamescope_available]
            if return_steam_cmd:
                gamescope_cmd = ['gamescope']
            if params.gamescope_fullscreen:
                gamescope_cmd.append('-f')
            if params.gamescope_borderless:
                gamescope_cmd.append('-b')
            if params.gamescope_scaling:
                gamescope_cmd.append('-n')
            if params.fsr:
                gamescope_cmd.append('-U')
                gamescope_cmd.append(f'--fsr-sharpness {params.fsr_sharpening_strength * 4}')
            if params.gamescope_fps > 0:
                gamescope_cmd.append(f'-r {params.gamescope_fps}')
            if params.gamescope_fps_no_focus > 0:
                gamescope_cmd.append(f'-o {params.gamescope_fps_no_focus}')
            if params.gamescope_game_width > 0:
                gamescope_cmd.append(f'-w {params.gamescope_game_width}')
            if params.gamescope_game_height > 0:
                gamescope_cmd.append(f'-h {params.gamescope_game_height}')
            if params.gamescope_window_width > 0:
                gamescope_cmd.append(f'-W {params.gamescope_window_width}')
            if params.gamescope_window_height > 0:
                gamescope_cmd.append(f'-H {params.gamescope_window_height}')
        return ' '.join(gamescope_cmd)

    def _vmtouch_preload(self):
        if False:
            while True:
                i = 10
        vmtouch_flags = '-t -v -l -d'
        vmtouch_file_size = ' -m 1024M'
        if self.command.find('C:\\') > 0:
            s = (self.cwd + '/' + self.command.split(' ')[-1].split('\\')[-1]).replace("'", '')
        else:
            s = self.command.split(' ')[-1]
        self.vmtouch_files = f"'{s}'"
        self.command = f'{vmtouch_available} {vmtouch_flags} {vmtouch_file_size} {self.vmtouch_files} && {self.command}'

    def _vmtouch_free(self):
        if False:
            while True:
                i = 10
        subprocess.Popen('kill $(pidof vmtouch)', shell=True, env=self.env, cwd=self.cwd)
        if not self.vmtouch_files:
            return
        vmtouch_flags = '-e -v'
        command = f'{vmtouch_available} {vmtouch_flags} {self.vmtouch_files}'
        subprocess.Popen(command, shell=True, env=self.env, cwd=self.cwd)

    def _get_sandbox_manager(self) -> SandboxManager:
        if False:
            for i in range(10):
                print('nop')
        return SandboxManager(envs=self.env, chdir=self.cwd, share_paths_rw=[ManagerUtils.get_bottle_path(self.config)], share_paths_ro=[Paths.runners, Paths.temp], share_net=self.config.Sandbox.share_net, share_sound=self.config.Sandbox.share_sound)

    def run(self) -> Result[Optional[str]]:
        if False:
            return 10
        '\n        Run command with pre-configured parameters\n\n        :return: `status` is True if command executed successfully,\n                 `data` may be available even if `status` is False.\n        '
        if None in [self.runner, self.env]:
            return Result(False, message='runner or env is not ready, Wine command terminated.')
        if vmtouch_available and self.config.Parameters.vmtouch and (not self.terminal):
            self._vmtouch_preload()
        sandbox = self._get_sandbox_manager() if self.config.Parameters.sandbox else None
        if self.terminal:
            if sandbox:
                return Result(status=TerminalUtils().execute(sandbox.get_cmd(self.command), self.env, self.colors, self.cwd))
            else:
                return Result(status=TerminalUtils().execute(self.command, self.env, self.colors, self.cwd))
        proc: subprocess.Popen[bytes]
        if sandbox:
            proc = sandbox.run(self.command)
        else:
            try:
                proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, shell=True, env=self.env, cwd=self.cwd)
            except FileNotFoundError:
                return Result(False, message='File not found')
        (stdout_data, _) = proc.communicate()
        if vmtouch_available and self.config.Parameters.vmtouch:
            self._vmtouch_free()
        codec = detect_encoding(stdout_data)
        rv: str
        try:
            rv = stdout_data.decode(codec)
        except (UnicodeDecodeError, LookupError, TypeError):
            logging.warning(f'stdout decoding failed')
            rv = str(stdout_data)[2:-1]
        if 'ShellExecuteEx' in rv:
            logging.warning('ShellExecuteEx exception seems occurred.')
            return Result(False, data=rv, message='ShellExecuteEx exception seems occurred.')
        return Result(True, data=rv)

    @staticmethod
    def __set_dxvk_nvapi_conf(bottle: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        TODO: This should be moved to a dedicated DXVKConf class when\n              we will provide a way to set the DXVK configuration.\n        '
        dxvk_conf = f'{bottle}/dxvk.conf'
        if not os.path.exists(dxvk_conf):
            with open(dxvk_conf, 'w') as f:
                f.write('dxgi.nvapiHack = False')
        else:
            with open(dxvk_conf, 'r') as f:
                lines = f.readlines()
            with open(dxvk_conf, 'w') as f:
                for line in lines:
                    if 'dxgi.nvapiHack' in line:
                        f.write('dxgi.nvapiHack = False\n')
                    else:
                        f.write(line)
        return dxvk_conf