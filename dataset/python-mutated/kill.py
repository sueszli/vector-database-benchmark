"""
This module provides the logic for `pyre kill`. Normally one would cleanly
shut down a pyre daemon using `pyre stop`, but if the daemon is not
responsive `pyre kill` will kill it right away.
"""
import logging
import os
import shutil
import signal
import psutil
from .. import daemon_socket, find_directories, frontend_configuration, identifiers
from . import commands, stop
LOG: logging.Logger = logging.getLogger(__name__)
PYRE_FIRE = "\n                                                    ',\n                                                   ,c:\n                                ',                ;cc:\n                              ,:l;             ';:lll:\n                            'cllc'            'clllll:\n                           ;loooc'           'cllllllc'\n                          ;looool,           :llllllll,       :,\n                         'looooool'         ;lollllllo:      ;ll;\n                         ;ooooooool,       'loooooloool;    ,clll:\n                         cdoddoooooo;      ;oooooooooool:  ;loooll:\n                         cddddddddodo;     cooooooooooooolloooooool;\n                         ;ddddddddddo;    'loooooooooooooooooooooooc'\n                          cdddddddddc     'ldddddooooooooooooooooooo,\n                           ,coodolc;       cddddddddddoooooooooooooo;\n                               '           ,oddddddddddddddddodooooo;\n                          ,::::::::::::,    :ddddddddddddddddddddddl'\n                          'lddddddddddxd:    :ddddddddddddddddddddd:\n                            ;odddddddddddl,   ;oxdddddddddddddddddl'\n                             'ldddddddddddo:   ,:ldxddxddddddddddl'\n                               :ddddddddddddl'    cdxxxxxxxdddddl'\n                                ,ldddddddddddo;    ,oxxxxxxxxxdc\n                                  :ddddddddddddc;'  'cdxxxxxxo;\n                                   ,ldddddddddddxo;   ;dxxxo:\n                                     cdddddddddddddc   'lo:\n                                      ;oddddddddddddo,\n                                       'cddddddddddddd:\n                                        ;odddddddddddddl,\n                                       :ddddddddddddddddd:\n                                     ,ldddddddddddddddddddl,\n                                    :odddddddddddddddddddddo:\n                                  'ldddddddddddddddddddddddddl'\n                                 ;odddddddddddl, ,ldddddddddddo;\n                               'cdddddddddddd:     :ddddddddddddc'\n                              ;odddddddddddo,       ,odddddddddddo;\n                             cddddddddddddc           cddddddddddddc\n                           ;oxddxddxddddo;             ;odxxxxddddxxo,\n                           ;:::::::::::;'               ';:::::::::::;\n"

def _kill_processes_by_name(name: str) -> None:
    if False:
        while True:
            i = 10
    for process in psutil.process_iter(attrs=['name']):
        if process.name() != name:
            continue
        pid_to_kill = process.pid
        if pid_to_kill == os.getpgid(os.getpid()):
            continue
        try:
            LOG.info(f'Killing process {name} with pid {pid_to_kill}.')
            os.kill(pid_to_kill, signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as exception:
            LOG.error(f'Failed to kill process {name} with pid {pid_to_kill} ' + f'due to exception {exception}')

def _kill_binary_processes(configuration: frontend_configuration.Base) -> None:
    if False:
        return 10
    LOG.warning('Force-killing all running pyre servers.')
    LOG.warning('Use `pyre servers stop` if you want to gracefully stop all running servers.')
    binary = configuration.get_binary_location(download_if_needed=False)
    if binary is not None:
        _kill_processes_by_name(str(binary))

def _kill_client_processes(configuration: frontend_configuration.Base) -> None:
    if False:
        while True:
            i = 10
    _kill_processes_by_name(find_directories.CLIENT_NAME)

def _delete_server_files(configuration: frontend_configuration.Base, flavor: identifiers.PyreFlavor) -> None:
    if False:
        print('Hello World!')
    socket_root = daemon_socket.get_default_socket_root()
    LOG.info(f'Deleting socket files and lock files under {socket_root}')
    for socket_path in daemon_socket.find_socket_files(socket_root):
        stop.remove_socket_if_exists(socket_path)
    log_directory = configuration.get_log_directory() / flavor.server_log_subdirectory()
    LOG.info(f'Deleting server logs under {log_directory}')
    try:
        shutil.rmtree(str(log_directory), ignore_errors=True)
    except OSError:
        pass

def _delete_caches(configuration: frontend_configuration.Base) -> None:
    if False:
        return 10
    dot_pyre_directory = configuration.get_dot_pyre_directory()
    resource_cache_directory = dot_pyre_directory / 'resource_cache'
    LOG.info(f'Deleting local binary and typeshed cache under {resource_cache_directory}')
    try:
        shutil.rmtree(str(resource_cache_directory), ignore_errors=True)
    except OSError:
        pass

def run(configuration: frontend_configuration.Base, with_fire: bool) -> commands.ExitCode:
    if False:
        print('Hello World!')
    _kill_binary_processes(configuration)
    _kill_client_processes(configuration)
    for flavor in [identifiers.PyreFlavor.CLASSIC, identifiers.PyreFlavor.CODE_NAVIGATION]:
        _delete_server_files(configuration, flavor)
    _delete_caches(configuration)
    if with_fire:
        LOG.warning('Note that `--with-fire` adds emphasis to `pyre kill` but does' + f' not affect its behavior.\n{PYRE_FIRE}')
    LOG.info('Done\n')
    return commands.ExitCode.SUCCESS