from __future__ import annotations
import os
import subprocess
import sys
from ansible.module_utils.common.text.converters import to_bytes

def has_respawned():
    if False:
        return 10
    return hasattr(sys.modules['__main__'], '_respawned')

def respawn_module(interpreter_path):
    if False:
        i = 10
        return i + 15
    '\n    Respawn the currently-running Ansible Python module under the specified Python interpreter.\n\n    Ansible modules that require libraries that are typically available only under well-known interpreters\n    (eg, ``yum``, ``apt``, ``dnf``) can use bespoke logic to determine the libraries they need are not\n    available, then call `respawn_module` to re-execute the current module under a different interpreter\n    and exit the current process when the new subprocess has completed. The respawned process inherits only\n    stdout/stderr from the current process.\n\n    Only a single respawn is allowed. ``respawn_module`` will fail on nested respawns. Modules are encouraged\n    to call `has_respawned()` to defensively guide behavior before calling ``respawn_module``, and to ensure\n    that the target interpreter exists, as ``respawn_module`` will not fail gracefully.\n\n    :arg interpreter_path: path to a Python interpreter to respawn the current module\n    '
    if has_respawned():
        raise Exception('module has already been respawned')
    payload = _create_payload()
    (stdin_read, stdin_write) = os.pipe()
    os.write(stdin_write, to_bytes(payload))
    os.close(stdin_write)
    rc = subprocess.call([interpreter_path, '--'], stdin=stdin_read)
    sys.exit(rc)

def probe_interpreters_for_module(interpreter_paths, module_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Probes a supplied list of Python interpreters, returning the first one capable of\n    importing the named module. This is useful when attempting to locate a "system\n    Python" where OS-packaged utility modules are located.\n\n    :arg interpreter_paths: iterable of paths to Python interpreters. The paths will be probed\n    in order, and the first path that exists and can successfully import the named module will\n    be returned (or ``None`` if probing fails for all supplied paths).\n    :arg module_name: fully-qualified Python module name to probe for (eg, ``selinux``)\n    '
    for interpreter_path in interpreter_paths:
        if not os.path.exists(interpreter_path):
            continue
        try:
            rc = subprocess.call([interpreter_path, '-c', 'import {0}'.format(module_name)])
            if rc == 0:
                return interpreter_path
        except Exception:
            continue
    return None

def _create_payload():
    if False:
        return 10
    from ansible.module_utils import basic
    smuggled_args = getattr(basic, '_ANSIBLE_ARGS')
    if not smuggled_args:
        raise Exception('unable to access ansible.module_utils.basic._ANSIBLE_ARGS (not launched by AnsiballZ?)')
    module_fqn = sys.modules['__main__']._module_fqn
    modlib_path = sys.modules['__main__']._modlib_path
    respawn_code_template = "\nimport runpy\nimport sys\n\nmodule_fqn = {module_fqn!r}\nmodlib_path = {modlib_path!r}\nsmuggled_args = {smuggled_args!r}\n\nif __name__ == '__main__':\n    sys.path.insert(0, modlib_path)\n\n    from ansible.module_utils import basic\n    basic._ANSIBLE_ARGS = smuggled_args\n\n    runpy.run_module(module_fqn, init_globals=dict(_respawned=True), run_name='__main__', alter_sys=True)\n    "
    respawn_code = respawn_code_template.format(module_fqn=module_fqn, modlib_path=modlib_path, smuggled_args=smuggled_args.strip())
    return respawn_code