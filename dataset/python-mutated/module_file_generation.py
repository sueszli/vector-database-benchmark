from typing import Optional, Set
from llnl.util import tty
import spack.config
import spack.modules
import spack.spec

def _for_each_enabled(spec: spack.spec.Spec, method_name: str, explicit: Optional[bool]=None) -> None:
    if False:
        i = 10
        return i + 15
    'Calls a method for each enabled module'
    set_names: Set[str] = set(spack.config.get('modules', {}).keys())
    for name in set_names:
        enabled = spack.config.get(f'modules:{name}:enable')
        if not enabled:
            tty.debug('NO MODULE WRITTEN: list of enabled module files is empty')
            continue
        for module_type in enabled:
            generator = spack.modules.module_types[module_type](spec, name, explicit)
            try:
                getattr(generator, method_name)()
            except RuntimeError as e:
                msg = 'cannot perform the requested {0} operation on module files'
                msg += ' [{1}]'
                tty.warn(msg.format(method_name, str(e)))

def post_install(spec, explicit: bool):
    if False:
        while True:
            i = 10
    import spack.environment as ev
    if ev.active_environment():
        return
    _for_each_enabled(spec, 'write', explicit)

def post_uninstall(spec):
    if False:
        i = 10
        return i + 15
    _for_each_enabled(spec, 'remove')

def post_env_write(env):
    if False:
        for i in range(10):
            print('nop')
    for spec in env.new_installs:
        _for_each_enabled(spec, 'write')