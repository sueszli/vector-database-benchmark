from __future__ import annotations
import ast
import base64
import datetime
import json
import os
import shlex
import time
import zipfile
import re
import pkgutil
from ast import AST, Import, ImportFrom
from io import BytesIO
from ansible.release import __version__, __author__
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.executor.interpreter_discovery import InterpreterDiscoveryRequiredError
from ansible.executor.powershell import module_manifest as ps_manifest
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.plugins.loader import module_utils_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, _nested_dict_get
from ansible.executor import action_write_locks
from ansible.utils.display import Display
from collections import namedtuple
import importlib.util
import importlib.machinery
display = Display()
ModuleUtilsProcessEntry = namedtuple('ModuleUtilsProcessEntry', ['name_parts', 'is_ambiguous', 'has_redirected_child', 'is_optional'])
REPLACER = b'#<<INCLUDE_ANSIBLE_MODULE_COMMON>>'
REPLACER_VERSION = b'"<<ANSIBLE_VERSION>>"'
REPLACER_COMPLEX = b'"<<INCLUDE_ANSIBLE_MODULE_COMPLEX_ARGS>>"'
REPLACER_WINDOWS = b'# POWERSHELL_COMMON'
REPLACER_JSONARGS = b'<<INCLUDE_ANSIBLE_MODULE_JSON_ARGS>>'
REPLACER_SELINUX = b'<<SELINUX_SPECIAL_FILESYSTEMS>>'
ENCODING_STRING = u'# -*- coding: utf-8 -*-'
b_ENCODING_STRING = b'# -*- coding: utf-8 -*-'
_MODULE_UTILS_PATH = os.path.join(os.path.dirname(__file__), '..', 'module_utils')
ANSIBALLZ_TEMPLATE = u'%(shebang)s\n%(coding)s\n_ANSIBALLZ_WRAPPER = True # For test-module.py script to tell this is a ANSIBALLZ_WRAPPER\n# This code is part of Ansible, but is an independent component.\n# The code in this particular templatable string, and this templatable string\n# only, is BSD licensed.  Modules which end up using this snippet, which is\n# dynamically combined together by Ansible still belong to the author of the\n# module, and they may assign their own license to the complete work.\n#\n# Copyright (c), James Cammarata, 2016\n# Copyright (c), Toshio Kuratomi, 2016\n#\n# Redistribution and use in source and binary forms, with or without modification,\n# are permitted provided that the following conditions are met:\n#\n#    * Redistributions of source code must retain the above copyright\n#      notice, this list of conditions and the following disclaimer.\n#    * Redistributions in binary form must reproduce the above copyright notice,\n#      this list of conditions and the following disclaimer in the documentation\n#      and/or other materials provided with the distribution.\n#\n# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND\n# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED\n# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.\n# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,\n# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,\n# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS\n# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT\n# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE\n# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\ndef _ansiballz_main():\n    import os\n    import os.path\n\n    # Access to the working directory is required by Python when using pipelining, as well as for the coverage module.\n    # Some platforms, such as macOS, may not allow querying the working directory when using become to drop privileges.\n    try:\n        os.getcwd()\n    except OSError:\n        try:\n            os.chdir(os.path.expanduser(\'~\'))\n        except OSError:\n            os.chdir(\'/\')\n\n%(rlimit)s\n\n    import sys\n    import __main__\n\n    # For some distros and python versions we pick up this script in the temporary\n    # directory.  This leads to problems when the ansible module masks a python\n    # library that another import needs.  We have not figured out what about the\n    # specific distros and python versions causes this to behave differently.\n    #\n    # Tested distros:\n    # Fedora23 with python3.4  Works\n    # Ubuntu15.10 with python2.7  Works\n    # Ubuntu15.10 with python3.4  Fails without this\n    # Ubuntu16.04.1 with python3.5  Fails without this\n    # To test on another platform:\n    # * use the copy module (since this shadows the stdlib copy module)\n    # * Turn off pipelining\n    # * Make sure that the destination file does not exist\n    # * ansible ubuntu16-test -m copy -a \'src=/etc/motd dest=/var/tmp/m\'\n    # This will traceback in shutil.  Looking at the complete traceback will show\n    # that shutil is importing copy which finds the ansible module instead of the\n    # stdlib module\n    scriptdir = None\n    try:\n        scriptdir = os.path.dirname(os.path.realpath(__main__.__file__))\n    except (AttributeError, OSError):\n        # Some platforms don\'t set __file__ when reading from stdin\n        # OSX raises OSError if using abspath() in a directory we don\'t have\n        # permission to read (realpath calls abspath)\n        pass\n\n    # Strip cwd from sys.path to avoid potential permissions issues\n    excludes = set((\'\', \'.\', scriptdir))\n    sys.path = [p for p in sys.path if p not in excludes]\n\n    import base64\n    import runpy\n    import shutil\n    import tempfile\n    import zipfile\n\n    if sys.version_info < (3,):\n        PY3 = False\n    else:\n        PY3 = True\n\n    ZIPDATA = %(zipdata)r\n\n    # Note: temp_path isn\'t needed once we switch to zipimport\n    def invoke_module(modlib_path, temp_path, json_params):\n        # When installed via setuptools (including python setup.py install),\n        # ansible may be installed with an easy-install.pth file.  That file\n        # may load the system-wide install of ansible rather than the one in\n        # the module.  sitecustomize is the only way to override that setting.\n        z = zipfile.ZipFile(modlib_path, mode=\'a\')\n\n        # py3: modlib_path will be text, py2: it\'s bytes.  Need bytes at the end\n        sitecustomize = u\'import sys\\nsys.path.insert(0,"%%s")\\n\' %% modlib_path\n        sitecustomize = sitecustomize.encode(\'utf-8\')\n        # Use a ZipInfo to work around zipfile limitation on hosts with\n        # clocks set to a pre-1980 year (for instance, Raspberry Pi)\n        zinfo = zipfile.ZipInfo()\n        zinfo.filename = \'sitecustomize.py\'\n        zinfo.date_time = %(date_time)s\n        z.writestr(zinfo, sitecustomize)\n        z.close()\n\n        # Put the zipped up module_utils we got from the controller first in the python path so that we\n        # can monkeypatch the right basic\n        sys.path.insert(0, modlib_path)\n\n        # Monkeypatch the parameters into basic\n        from ansible.module_utils import basic\n        basic._ANSIBLE_ARGS = json_params\n%(coverage)s\n        # Run the module!  By importing it as \'__main__\', it thinks it is executing as a script\n        runpy.run_module(mod_name=%(module_fqn)r, init_globals=dict(_module_fqn=%(module_fqn)r, _modlib_path=modlib_path),\n                         run_name=\'__main__\', alter_sys=True)\n\n        # Ansible modules must exit themselves\n        print(\'{"msg": "New-style module did not handle its own exit", "failed": true}\')\n        sys.exit(1)\n\n    def debug(command, zipped_mod, json_params):\n        # The code here normally doesn\'t run.  It\'s only used for debugging on the\n        # remote machine.\n        #\n        # The subcommands in this function make it easier to debug ansiballz\n        # modules.  Here\'s the basic steps:\n        #\n        # Run ansible with the environment variable: ANSIBLE_KEEP_REMOTE_FILES=1 and -vvv\n        # to save the module file remotely::\n        #   $ ANSIBLE_KEEP_REMOTE_FILES=1 ansible host1 -m ping -a \'data=october\' -vvv\n        #\n        # Part of the verbose output will tell you where on the remote machine the\n        # module was written to::\n        #   [...]\n        #   <host1> SSH: EXEC ssh -C -q -o ControlMaster=auto -o ControlPersist=60s -o KbdInteractiveAuthentication=no -o\n        #   PreferredAuthentications=gssapi-with-mic,gssapi-keyex,hostbased,publickey -o PasswordAuthentication=no -o ConnectTimeout=10 -o\n        #   ControlPath=/home/badger/.ansible/cp/ansible-ssh-%%h-%%p-%%r -tt rhel7 \'/bin/sh -c \'"\'"\'LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8\n        #   LC_MESSAGES=en_US.UTF-8 /usr/bin/python /home/badger/.ansible/tmp/ansible-tmp-1461173013.93-9076457629738/ping\'"\'"\'\'\n        #   [...]\n        #\n        # Login to the remote machine and run the module file via from the previous\n        # step with the explode subcommand to extract the module payload into\n        # source files::\n        #   $ ssh host1\n        #   $ /usr/bin/python /home/badger/.ansible/tmp/ansible-tmp-1461173013.93-9076457629738/ping explode\n        #   Module expanded into:\n        #   /home/badger/.ansible/tmp/ansible-tmp-1461173408.08-279692652635227/ansible\n        #\n        # You can now edit the source files to instrument the code or experiment with\n        # different parameter values.  When you\'re ready to run the code you\'ve modified\n        # (instead of the code from the actual zipped module), use the execute subcommand like this::\n        #   $ /usr/bin/python /home/badger/.ansible/tmp/ansible-tmp-1461173013.93-9076457629738/ping execute\n\n        # Okay to use __file__ here because we\'re running from a kept file\n        basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), \'debug_dir\')\n        args_path = os.path.join(basedir, \'args\')\n\n        if command == \'explode\':\n            # transform the ZIPDATA into an exploded directory of code and then\n            # print the path to the code.  This is an easy way for people to look\n            # at the code on the remote machine for debugging it in that\n            # environment\n            z = zipfile.ZipFile(zipped_mod)\n            for filename in z.namelist():\n                if filename.startswith(\'/\'):\n                    raise Exception(\'Something wrong with this module zip file: should not contain absolute paths\')\n\n                dest_filename = os.path.join(basedir, filename)\n                if dest_filename.endswith(os.path.sep) and not os.path.exists(dest_filename):\n                    os.makedirs(dest_filename)\n                else:\n                    directory = os.path.dirname(dest_filename)\n                    if not os.path.exists(directory):\n                        os.makedirs(directory)\n                    f = open(dest_filename, \'wb\')\n                    f.write(z.read(filename))\n                    f.close()\n\n            # write the args file\n            f = open(args_path, \'wb\')\n            f.write(json_params)\n            f.close()\n\n            print(\'Module expanded into:\')\n            print(\'%%s\' %% basedir)\n            exitcode = 0\n\n        elif command == \'execute\':\n            # Execute the exploded code instead of executing the module from the\n            # embedded ZIPDATA.  This allows people to easily run their modified\n            # code on the remote machine to see how changes will affect it.\n\n            # Set pythonpath to the debug dir\n            sys.path.insert(0, basedir)\n\n            # read in the args file which the user may have modified\n            with open(args_path, \'rb\') as f:\n                json_params = f.read()\n\n            # Monkeypatch the parameters into basic\n            from ansible.module_utils import basic\n            basic._ANSIBLE_ARGS = json_params\n\n            # Run the module!  By importing it as \'__main__\', it thinks it is executing as a script\n            runpy.run_module(mod_name=%(module_fqn)r, init_globals=None, run_name=\'__main__\', alter_sys=True)\n\n            # Ansible modules must exit themselves\n            print(\'{"msg": "New-style module did not handle its own exit", "failed": true}\')\n            sys.exit(1)\n\n        else:\n            print(\'WARNING: Unknown debug command.  Doing nothing.\')\n            exitcode = 0\n\n        return exitcode\n\n    #\n    # See comments in the debug() method for information on debugging\n    #\n\n    ANSIBALLZ_PARAMS = %(params)s\n    if PY3:\n        ANSIBALLZ_PARAMS = ANSIBALLZ_PARAMS.encode(\'utf-8\')\n    try:\n        # There\'s a race condition with the controller removing the\n        # remote_tmpdir and this module executing under async.  So we cannot\n        # store this in remote_tmpdir (use system tempdir instead)\n        # Only need to use [ansible_module]_payload_ in the temp_path until we move to zipimport\n        # (this helps ansible-test produce coverage stats)\n        temp_path = tempfile.mkdtemp(prefix=\'ansible_\' + %(ansible_module)r + \'_payload_\')\n\n        zipped_mod = os.path.join(temp_path, \'ansible_\' + %(ansible_module)r + \'_payload.zip\')\n\n        with open(zipped_mod, \'wb\') as modlib:\n            modlib.write(base64.b64decode(ZIPDATA))\n\n        if len(sys.argv) == 2:\n            exitcode = debug(sys.argv[1], zipped_mod, ANSIBALLZ_PARAMS)\n        else:\n            # Note: temp_path isn\'t needed once we switch to zipimport\n            invoke_module(zipped_mod, temp_path, ANSIBALLZ_PARAMS)\n    finally:\n        try:\n            shutil.rmtree(temp_path)\n        except (NameError, OSError):\n            # tempdir creation probably failed\n            pass\n    sys.exit(exitcode)\n\nif __name__ == \'__main__\':\n    _ansiballz_main()\n'
ANSIBALLZ_COVERAGE_TEMPLATE = '\n        os.environ[\'COVERAGE_FILE\'] = %(coverage_output)r + \'=python-%%s=coverage\' %% \'.\'.join(str(v) for v in sys.version_info[:2])\n\n        import atexit\n\n        try:\n            import coverage\n        except ImportError:\n            print(\'{"msg": "Could not import `coverage` module.", "failed": true}\')\n            sys.exit(1)\n\n        cov = coverage.Coverage(config_file=%(coverage_config)r)\n\n        def atexit_coverage():\n            cov.stop()\n            cov.save()\n\n        atexit.register(atexit_coverage)\n\n        cov.start()\n'
ANSIBALLZ_COVERAGE_CHECK_TEMPLATE = '\n        try:\n            if PY3:\n                import importlib.util\n                if importlib.util.find_spec(\'coverage\') is None:\n                    raise ImportError\n            else:\n                import imp\n                imp.find_module(\'coverage\')\n        except ImportError:\n            print(\'{"msg": "Could not find `coverage` module.", "failed": true}\')\n            sys.exit(1)\n'
ANSIBALLZ_RLIMIT_TEMPLATE = '\n    import resource\n\n    existing_soft, existing_hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n\n    # adjust soft limit subject to existing hard limit\n    requested_soft = min(existing_hard, %(rlimit_nofile)d)\n\n    if requested_soft != existing_soft:\n        try:\n            resource.setrlimit(resource.RLIMIT_NOFILE, (requested_soft, existing_hard))\n        except ValueError:\n            # some platforms (eg macOS) lie about their hard limit\n            pass\n'

def _strip_comments(source):
    if False:
        return 10
    buf = []
    for line in source.splitlines():
        l = line.strip()
        if not l or l.startswith(u'#'):
            continue
        buf.append(line)
    return u'\n'.join(buf)
if C.DEFAULT_KEEP_REMOTE_FILES:
    ACTIVE_ANSIBALLZ_TEMPLATE = ANSIBALLZ_TEMPLATE
else:
    ACTIVE_ANSIBALLZ_TEMPLATE = _strip_comments(ANSIBALLZ_TEMPLATE)
site_packages = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CORE_LIBRARY_PATH_RE = re.compile('%s/(?P<path>ansible/modules/.*)\\.(py|ps1)$' % re.escape(site_packages))
COLLECTION_PATH_RE = re.compile('/(?P<path>ansible_collections/[^/]+/[^/]+/plugins/modules/.*)\\.(py|ps1)$')
NEW_STYLE_PYTHON_MODULE_RE = re.compile(b'(?:from +\\.{2,} *module_utils.* +import |from +ansible_collections\\.[^.]+\\.[^.]+\\.plugins\\.module_utils.* +import |import +ansible_collections\\.[^.]+\\.[^.]+\\.plugins\\.module_utils.*|from +ansible\\.module_utils.* +import |import +ansible\\.module_utils\\.)')

class ModuleDepFinder(ast.NodeVisitor):

    def __init__(self, module_fqn, tree, is_pkg_init=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Walk the ast tree for the python module.\n        :arg module_fqn: The fully qualified name to reach this module in dotted notation.\n            example: ansible.module_utils.basic\n        :arg is_pkg_init: Inform the finder it's looking at a package init (eg __init__.py) to allow\n            relative import expansion to use the proper package level without having imported it locally first.\n\n        Save submodule[.submoduleN][.identifier] into self.submodules\n        when they are from ansible.module_utils or ansible_collections packages\n\n        self.submodules will end up with tuples like:\n          - ('ansible', 'module_utils', 'basic',)\n          - ('ansible', 'module_utils', 'urls', 'fetch_url')\n          - ('ansible', 'module_utils', 'database', 'postgres')\n          - ('ansible', 'module_utils', 'database', 'postgres', 'quote')\n          - ('ansible', 'module_utils', 'database', 'postgres', 'quote')\n          - ('ansible_collections', 'my_ns', 'my_col', 'plugins', 'module_utils', 'foo')\n\n        It's up to calling code to determine whether the final element of the\n        tuple are module names or something else (function, class, or variable names)\n        .. seealso:: :python3:class:`ast.NodeVisitor`\n        "
        super(ModuleDepFinder, self).__init__(*args, **kwargs)
        self._tree = tree
        self.submodules = set()
        self.optional_imports = set()
        self.module_fqn = module_fqn
        self.is_pkg_init = is_pkg_init
        self._visit_map = {Import: self.visit_Import, ImportFrom: self.visit_ImportFrom}
        self.visit(tree)

    def generic_visit(self, node):
        if False:
            i = 10
            return i + 15
        'Overridden ``generic_visit`` that makes some assumptions about our\n        use case, and improves performance by calling visitors directly instead\n        of calling ``visit`` to offload calling visitors.\n        '
        generic_visit = self.generic_visit
        visit_map = self._visit_map
        for (field, value) in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, (Import, ImportFrom)):
                        item.parent = node
                        visit_map[item.__class__](item)
                    elif isinstance(item, AST):
                        generic_visit(item)
    visit = generic_visit

    def visit_Import(self, node):
        if False:
            return 10
        '\n        Handle import ansible.module_utils.MODLIB[.MODLIBn] [as asname]\n\n        We save these as interesting submodules when the imported library is in ansible.module_utils\n        or ansible.collections\n        '
        for alias in node.names:
            if alias.name.startswith('ansible.module_utils.') or alias.name.startswith('ansible_collections.'):
                py_mod = tuple(alias.name.split('.'))
                self.submodules.add(py_mod)
                if node.parent != self._tree:
                    self.optional_imports.add(py_mod)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if False:
            while True:
                i = 10
        '\n        Handle from ansible.module_utils.MODLIB import [.MODLIBn] [as asname]\n\n        Also has to handle relative imports\n\n        We save these as interesting submodules when the imported library is in ansible.module_utils\n        or ansible.collections\n        '
        if node.level > 0:
            level_slice_offset = -node.level + 1 or None if self.is_pkg_init else -node.level
            if self.module_fqn:
                parts = tuple(self.module_fqn.split('.'))
                if node.module:
                    node_module = '.'.join(parts[:level_slice_offset] + (node.module,))
                else:
                    node_module = '.'.join(parts[:level_slice_offset])
            else:
                node_module = node.module
        else:
            node_module = node.module
        py_mod = None
        if node.names[0].name == '_six':
            self.submodules.add(('_six',))
        elif node_module.startswith('ansible.module_utils'):
            py_mod = tuple(node_module.split('.'))
        elif node_module.startswith('ansible_collections.'):
            if node_module.endswith('plugins.module_utils') or '.plugins.module_utils.' in node_module:
                py_mod = tuple(node_module.split('.'))
            else:
                pass
        if py_mod:
            for alias in node.names:
                self.submodules.add(py_mod + (alias.name,))
                if node.parent != self._tree:
                    self.optional_imports.add(py_mod + (alias.name,))
        self.generic_visit(node)

def _slurp(path):
    if False:
        return 10
    if not os.path.exists(path):
        raise AnsibleError('imported module support code does not exist at %s' % os.path.abspath(path))
    with open(path, 'rb') as fd:
        data = fd.read()
    return data

def _get_shebang(interpreter, task_vars, templar, args=tuple(), remote_is_local=False):
    if False:
        while True:
            i = 10
    '\n      Handles the different ways ansible allows overriding the shebang target for a module.\n    '
    interpreter_name = os.path.basename(interpreter).strip()
    interpreter_config = u'ansible_%s_interpreter' % interpreter_name
    interpreter_config_key = 'INTERPRETER_%s' % interpreter_name.upper()
    interpreter_out = None
    if interpreter_name == 'python':
        if remote_is_local:
            interpreter_out = task_vars['ansible_playbook_python']
        elif C.config.get_configuration_definition(interpreter_config_key):
            interpreter_from_config = C.config.get_config_value(interpreter_config_key, variables=task_vars)
            interpreter_out = templar.template(interpreter_from_config.strip())
            if not interpreter_out or interpreter_out in ['auto', 'auto_legacy', 'auto_silent', 'auto_legacy_silent']:
                discovered_interpreter_config = u'discovered_interpreter_%s' % interpreter_name
                facts_from_task_vars = task_vars.get('ansible_facts', {})
                if discovered_interpreter_config not in facts_from_task_vars:
                    raise InterpreterDiscoveryRequiredError('interpreter discovery needed', interpreter_name=interpreter_name, discovery_mode=interpreter_out)
                else:
                    interpreter_out = facts_from_task_vars[discovered_interpreter_config]
        else:
            raise InterpreterDiscoveryRequiredError('interpreter discovery required', interpreter_name=interpreter_name, discovery_mode='auto_legacy')
    elif interpreter_config in task_vars:
        interpreter_out = templar.template(task_vars.get(interpreter_config).strip())
    if not interpreter_out:
        interpreter_out = interpreter
    shebang = u'#!{0}'.format(interpreter_out)
    if args:
        shebang = shebang + u' ' + u' '.join(args)
    return (shebang, interpreter_out)

class ModuleUtilLocatorBase:

    def __init__(self, fq_name_parts, is_ambiguous=False, child_is_redirected=False, is_optional=False):
        if False:
            return 10
        self._is_ambiguous = is_ambiguous
        self._child_is_redirected = child_is_redirected
        self._is_optional = is_optional
        self.found = False
        self.redirected = False
        self.fq_name_parts = fq_name_parts
        self.source_code = ''
        self.output_path = ''
        self.is_package = False
        self._collection_name = None
        if is_ambiguous and len(self._get_module_utils_remainder_parts(fq_name_parts)) > 1:
            self.candidate_names = [fq_name_parts, fq_name_parts[:-1]]
        else:
            self.candidate_names = [fq_name_parts]

    @property
    def candidate_names_joined(self):
        if False:
            i = 10
            return i + 15
        return ['.'.join(n) for n in self.candidate_names]

    def _handle_redirect(self, name_parts):
        if False:
            while True:
                i = 10
        module_utils_relative_parts = self._get_module_utils_remainder_parts(name_parts)
        if not module_utils_relative_parts:
            return False
        try:
            collection_metadata = _get_collection_metadata(self._collection_name)
        except ValueError as ve:
            if self._is_optional:
                return False
            raise AnsibleError('error processing module_util {0} loading redirected collection {1}: {2}'.format('.'.join(name_parts), self._collection_name, to_native(ve)))
        routing_entry = _nested_dict_get(collection_metadata, ['plugin_routing', 'module_utils', '.'.join(module_utils_relative_parts)])
        if not routing_entry:
            return False
        dep_or_ts = routing_entry.get('tombstone')
        removed = dep_or_ts is not None
        if not removed:
            dep_or_ts = routing_entry.get('deprecation')
        if dep_or_ts:
            removal_date = dep_or_ts.get('removal_date')
            removal_version = dep_or_ts.get('removal_version')
            warning_text = dep_or_ts.get('warning_text')
            msg = 'module_util {0} has been removed'.format('.'.join(name_parts))
            if warning_text:
                msg += ' ({0})'.format(warning_text)
            else:
                msg += '.'
            display.deprecated(msg, removal_version, removed, removal_date, self._collection_name)
        if 'redirect' in routing_entry:
            self.redirected = True
            source_pkg = '.'.join(name_parts)
            self.is_package = True
            redirect_target_pkg = routing_entry['redirect']
            if not redirect_target_pkg.startswith('ansible_collections'):
                split_fqcn = redirect_target_pkg.split('.')
                if len(split_fqcn) < 3:
                    raise Exception('invalid redirect for {0}: {1}'.format(source_pkg, redirect_target_pkg))
                redirect_target_pkg = 'ansible_collections.{0}.{1}.plugins.module_utils.{2}'.format(split_fqcn[0], split_fqcn[1], '.'.join(split_fqcn[2:]))
            display.vvv('redirecting module_util {0} to {1}'.format(source_pkg, redirect_target_pkg))
            self.source_code = self._generate_redirect_shim_source(source_pkg, redirect_target_pkg)
            return True
        return False

    def _get_module_utils_remainder_parts(self, name_parts):
        if False:
            print('Hello World!')
        return []

    def _get_module_utils_remainder(self, name_parts):
        if False:
            return 10
        return '.'.join(self._get_module_utils_remainder_parts(name_parts))

    def _find_module(self, name_parts):
        if False:
            while True:
                i = 10
        return False

    def _locate(self, redirect_first=True):
        if False:
            for i in range(10):
                print('nop')
        for candidate_name_parts in self.candidate_names:
            if redirect_first and self._handle_redirect(candidate_name_parts):
                break
            if self._find_module(candidate_name_parts):
                break
            if not redirect_first and self._handle_redirect(candidate_name_parts):
                break
        else:
            if self._child_is_redirected:
                self.is_package = True
                self.source_code = ''
            else:
                return
        if self.is_package:
            path_parts = candidate_name_parts + ('__init__',)
        else:
            path_parts = candidate_name_parts
        self.found = True
        self.output_path = os.path.join(*path_parts) + '.py'
        self.fq_name_parts = candidate_name_parts

    def _generate_redirect_shim_source(self, fq_source_module, fq_target_module):
        if False:
            for i in range(10):
                print('nop')
        return "\nimport sys\nimport {1} as mod\n\nsys.modules['{0}'] = mod\n".format(fq_source_module, fq_target_module)

class LegacyModuleUtilLocator(ModuleUtilLocatorBase):

    def __init__(self, fq_name_parts, is_ambiguous=False, mu_paths=None, child_is_redirected=False):
        if False:
            while True:
                i = 10
        super(LegacyModuleUtilLocator, self).__init__(fq_name_parts, is_ambiguous, child_is_redirected)
        if fq_name_parts[0:2] != ('ansible', 'module_utils'):
            raise Exception('this class can only locate from ansible.module_utils, got {0}'.format(fq_name_parts))
        if fq_name_parts[2] == 'six':
            fq_name_parts = ('ansible', 'module_utils', 'six')
            self.candidate_names = [fq_name_parts]
        self._mu_paths = mu_paths
        self._collection_name = 'ansible.builtin'
        self._locate(redirect_first=False)

    def _get_module_utils_remainder_parts(self, name_parts):
        if False:
            return 10
        return name_parts[2:]

    def _find_module(self, name_parts):
        if False:
            while True:
                i = 10
        rel_name_parts = self._get_module_utils_remainder_parts(name_parts)
        if len(rel_name_parts) == 1:
            paths = self._mu_paths
        else:
            paths = [os.path.join(p, *rel_name_parts[:-1]) for p in self._mu_paths]
        self._info = info = importlib.machinery.PathFinder.find_spec('.'.join(name_parts), paths)
        if info is not None and os.path.splitext(info.origin)[1] in importlib.machinery.SOURCE_SUFFIXES:
            self.is_package = info.origin.endswith('/__init__.py')
            path = info.origin
        else:
            return False
        self.source_code = _slurp(path)
        return True

class CollectionModuleUtilLocator(ModuleUtilLocatorBase):

    def __init__(self, fq_name_parts, is_ambiguous=False, child_is_redirected=False, is_optional=False):
        if False:
            i = 10
            return i + 15
        super(CollectionModuleUtilLocator, self).__init__(fq_name_parts, is_ambiguous, child_is_redirected, is_optional)
        if fq_name_parts[0] != 'ansible_collections':
            raise Exception('CollectionModuleUtilLocator can only locate from ansible_collections, got {0}'.format(fq_name_parts))
        elif len(fq_name_parts) >= 6 and fq_name_parts[3:5] != ('plugins', 'module_utils'):
            raise Exception('CollectionModuleUtilLocator can only locate below ansible_collections.(ns).(coll).plugins.module_utils, got {0}'.format(fq_name_parts))
        self._collection_name = '.'.join(fq_name_parts[1:3])
        self._locate()

    def _find_module(self, name_parts):
        if False:
            for i in range(10):
                print('nop')
        if len(name_parts) < 6:
            self.source_code = ''
            self.is_package = True
            return True
        collection_pkg_name = '.'.join(name_parts[0:3])
        resource_base_path = os.path.join(*name_parts[3:])
        src = None
        try:
            src = pkgutil.get_data(collection_pkg_name, to_native(os.path.join(resource_base_path, '__init__.py')))
        except ImportError:
            pass
        if src is not None:
            self.is_package = True
        else:
            try:
                src = pkgutil.get_data(collection_pkg_name, to_native(resource_base_path + '.py'))
            except ImportError:
                pass
        if src is None:
            return False
        self.source_code = src
        return True

    def _get_module_utils_remainder_parts(self, name_parts):
        if False:
            return 10
        return name_parts[5:]

def _make_zinfo(filename, date_time, zf=None):
    if False:
        while True:
            i = 10
    zinfo = zipfile.ZipInfo(filename=filename, date_time=date_time)
    if zf:
        zinfo.compress_type = zf.compression
    return zinfo

def recursive_finder(name, module_fqn, module_data, zf, date_time=None):
    if False:
        return 10
    "\n    Using ModuleDepFinder, make sure we have all of the module_utils files that\n    the module and its module_utils files needs. (no longer actually recursive)\n    :arg name: Name of the python module we're examining\n    :arg module_fqn: Fully qualified name of the python module we're scanning\n    :arg module_data: string Python code of the module we're scanning\n    :arg zf: An open :python:class:`zipfile.ZipFile` object that holds the Ansible module payload\n        which we're assembling\n    "
    if date_time is None:
        date_time = time.gmtime()[:6]
    py_module_cache = {('ansible',): (b'from pkgutil import extend_path\n__path__=extend_path(__path__,__name__)\n__version__="' + to_bytes(__version__) + b'"\n__author__="' + to_bytes(__author__) + b'"\n', 'ansible/__init__.py'), ('ansible', 'module_utils'): (b'from pkgutil import extend_path\n__path__=extend_path(__path__,__name__)\n', 'ansible/module_utils/__init__.py')}
    module_utils_paths = [p for p in module_utils_loader._get_paths(subdirs=False) if os.path.isdir(p)]
    module_utils_paths.append(_MODULE_UTILS_PATH)
    try:
        tree = compile(module_data, '<unknown>', 'exec', ast.PyCF_ONLY_AST)
    except (SyntaxError, IndentationError) as e:
        raise AnsibleError('Unable to import %s due to %s' % (name, e.msg))
    finder = ModuleDepFinder(module_fqn, tree)
    modules_to_process = [ModuleUtilsProcessEntry(m, True, False, is_optional=m in finder.optional_imports) for m in finder.submodules]
    modules_to_process.append(ModuleUtilsProcessEntry(('ansible', 'module_utils', 'basic'), False, False, is_optional=False))
    while modules_to_process:
        modules_to_process.sort()
        (py_module_name, is_ambiguous, child_is_redirected, is_optional) = modules_to_process.pop(0)
        if py_module_name in py_module_cache:
            continue
        if py_module_name[0:2] == ('ansible', 'module_utils'):
            module_info = LegacyModuleUtilLocator(py_module_name, is_ambiguous=is_ambiguous, mu_paths=module_utils_paths, child_is_redirected=child_is_redirected)
        elif py_module_name[0] == 'ansible_collections':
            module_info = CollectionModuleUtilLocator(py_module_name, is_ambiguous=is_ambiguous, child_is_redirected=child_is_redirected, is_optional=is_optional)
        else:
            display.warning('ModuleDepFinder improperly found a non-module_utils import %s' % [py_module_name])
            continue
        if not module_info.found:
            if is_optional:
                continue
            msg = 'Could not find imported module support code for {0}.  Looked for ({1})'.format(module_fqn, module_info.candidate_names_joined)
            raise AnsibleError(msg)
        if module_info.fq_name_parts in py_module_cache:
            continue
        try:
            tree = compile(module_info.source_code, '<unknown>', 'exec', ast.PyCF_ONLY_AST)
        except (SyntaxError, IndentationError) as e:
            raise AnsibleError('Unable to import %s due to %s' % (module_info.fq_name_parts, e.msg))
        finder = ModuleDepFinder('.'.join(module_info.fq_name_parts), tree, module_info.is_package)
        modules_to_process.extend((ModuleUtilsProcessEntry(m, True, False, is_optional=m in finder.optional_imports) for m in finder.submodules if m not in py_module_cache))
        py_module_cache[module_info.fq_name_parts] = (module_info.source_code, module_info.output_path)
        accumulated_pkg_name = []
        for pkg in module_info.fq_name_parts[:-1]:
            accumulated_pkg_name.append(pkg)
            normalized_name = tuple(accumulated_pkg_name)
            if normalized_name not in py_module_cache:
                modules_to_process.append(ModuleUtilsProcessEntry(normalized_name, False, module_info.redirected, is_optional=is_optional))
    for py_module_name in py_module_cache:
        py_module_file_name = py_module_cache[py_module_name][1]
        zf.writestr(_make_zinfo(py_module_file_name, date_time, zf=zf), py_module_cache[py_module_name][0])
        mu_file = to_text(py_module_file_name, errors='surrogate_or_strict')
        display.vvvvv('Including module_utils file %s' % mu_file)

def _is_binary(b_module_data):
    if False:
        i = 10
        return i + 15
    textchars = bytearray(set([7, 8, 9, 10, 12, 13, 27]) | set(range(32, 256)) - set([127]))
    start = b_module_data[:1024]
    return bool(start.translate(None, textchars))

def _get_ansible_module_fqn(module_path):
    if False:
        return 10
    "\n    Get the fully qualified name for an ansible module based on its pathname\n\n    remote_module_fqn is the fully qualified name.  Like ansible.modules.system.ping\n    Or ansible_collections.Namespace.Collection_name.plugins.modules.ping\n    .. warning:: This function is for ansible modules only.  It won't work for other things\n        (non-module plugins, etc)\n    "
    remote_module_fqn = None
    match = CORE_LIBRARY_PATH_RE.search(module_path)
    if not match:
        match = COLLECTION_PATH_RE.search(module_path)
    if match:
        path = match.group('path')
        if '.' in path:
            raise ValueError('Module name (or path) was not a valid python identifier')
        remote_module_fqn = '.'.join(path.split('/'))
    else:
        raise ValueError("Unable to determine module's fully qualified name")
    return remote_module_fqn

def _add_module_to_zip(zf, date_time, remote_module_fqn, b_module_data):
    if False:
        return 10
    'Add a module from ansible or from an ansible collection into the module zip'
    module_path_parts = remote_module_fqn.split('.')
    module_path = '/'.join(module_path_parts) + '.py'
    zf.writestr(_make_zinfo(module_path, date_time, zf=zf), b_module_data)
    if module_path_parts[0] == 'ansible':
        start = 2
        existing_paths = frozenset()
    else:
        start = 1
        existing_paths = frozenset(zf.namelist())
    for idx in range(start, len(module_path_parts)):
        package_path = '/'.join(module_path_parts[:idx]) + '/__init__.py'
        if package_path in existing_paths:
            continue
        zf.writestr(_make_zinfo(package_path, date_time, zf=zf), b'')

def _find_module_utils(module_name, b_module_data, module_path, module_args, task_vars, templar, module_compression, async_timeout, become, become_method, become_user, become_password, become_flags, environment, remote_is_local=False):
    if False:
        while True:
            i = 10
    "\n    Given the source of the module, convert it to a Jinja2 template to insert\n    module code and return whether it's a new or old style module.\n    "
    module_substyle = module_style = 'old'
    if _is_binary(b_module_data):
        module_substyle = module_style = 'binary'
    elif REPLACER in b_module_data:
        module_style = 'new'
        module_substyle = 'python'
        b_module_data = b_module_data.replace(REPLACER, b'from ansible.module_utils.basic import *')
    elif NEW_STYLE_PYTHON_MODULE_RE.search(b_module_data):
        module_style = 'new'
        module_substyle = 'python'
    elif REPLACER_WINDOWS in b_module_data:
        module_style = 'new'
        module_substyle = 'powershell'
        b_module_data = b_module_data.replace(REPLACER_WINDOWS, b'#Requires -Module Ansible.ModuleUtils.Legacy')
    elif re.search(b'#Requires -Module', b_module_data, re.IGNORECASE) or re.search(b'#Requires -Version', b_module_data, re.IGNORECASE) or re.search(b'#AnsibleRequires -OSVersion', b_module_data, re.IGNORECASE) or re.search(b'#AnsibleRequires -Powershell', b_module_data, re.IGNORECASE) or re.search(b'#AnsibleRequires -CSharpUtil', b_module_data, re.IGNORECASE):
        module_style = 'new'
        module_substyle = 'powershell'
    elif REPLACER_JSONARGS in b_module_data:
        module_style = 'new'
        module_substyle = 'jsonargs'
    elif b'WANT_JSON' in b_module_data:
        module_substyle = module_style = 'non_native_want_json'
    shebang = None
    if module_style in ('old', 'non_native_want_json', 'binary'):
        return (b_module_data, module_style, shebang)
    output = BytesIO()
    try:
        remote_module_fqn = _get_ansible_module_fqn(module_path)
    except ValueError:
        display.debug('ANSIBALLZ: Could not determine module FQN')
        remote_module_fqn = 'ansible.modules.%s' % module_name
    if module_substyle == 'python':
        date_time = time.gmtime()[:6]
        if date_time[0] < 1980:
            date_string = datetime.datetime(*date_time, tzinfo=datetime.timezone.utc).strftime('%c')
            raise AnsibleError(f'Cannot create zipfile due to pre-1980 configured date: {date_string}')
        params = dict(ANSIBLE_MODULE_ARGS=module_args)
        try:
            python_repred_params = repr(json.dumps(params, cls=AnsibleJSONEncoder, vault_to_text=True))
        except TypeError as e:
            raise AnsibleError('Unable to pass options to module, they must be JSON serializable: %s' % to_native(e))
        try:
            compression_method = getattr(zipfile, module_compression)
        except AttributeError:
            display.warning(u'Bad module compression string specified: %s.  Using ZIP_STORED (no compression)' % module_compression)
            compression_method = zipfile.ZIP_STORED
        lookup_path = os.path.join(C.DEFAULT_LOCAL_TMP, 'ansiballz_cache')
        cached_module_filename = os.path.join(lookup_path, '%s-%s' % (remote_module_fqn, module_compression))
        zipdata = None
        if os.path.exists(cached_module_filename):
            display.debug('ANSIBALLZ: using cached module: %s' % cached_module_filename)
            with open(cached_module_filename, 'rb') as module_data:
                zipdata = module_data.read()
        else:
            if module_name in action_write_locks.action_write_locks:
                display.debug('ANSIBALLZ: Using lock for %s' % module_name)
                lock = action_write_locks.action_write_locks[module_name]
            else:
                display.debug('ANSIBALLZ: Using generic lock for %s' % module_name)
                lock = action_write_locks.action_write_locks[None]
            display.debug('ANSIBALLZ: Acquiring lock')
            with lock:
                display.debug('ANSIBALLZ: Lock acquired: %s' % id(lock))
                if not os.path.exists(cached_module_filename):
                    display.debug('ANSIBALLZ: Creating module')
                    zipoutput = BytesIO()
                    zf = zipfile.ZipFile(zipoutput, mode='w', compression=compression_method)
                    recursive_finder(module_name, remote_module_fqn, b_module_data, zf, date_time)
                    display.debug('ANSIBALLZ: Writing module into payload')
                    _add_module_to_zip(zf, date_time, remote_module_fqn, b_module_data)
                    zf.close()
                    zipdata = base64.b64encode(zipoutput.getvalue())
                    if not os.path.exists(lookup_path):
                        try:
                            os.makedirs(lookup_path)
                        except OSError:
                            if not os.path.exists(lookup_path):
                                raise
                    display.debug('ANSIBALLZ: Writing module')
                    with open(cached_module_filename + '-part', 'wb') as f:
                        f.write(zipdata)
                    display.debug('ANSIBALLZ: Renaming module')
                    os.rename(cached_module_filename + '-part', cached_module_filename)
                    display.debug('ANSIBALLZ: Done creating module')
            if zipdata is None:
                display.debug('ANSIBALLZ: Reading module after lock')
                try:
                    with open(cached_module_filename, 'rb') as f:
                        zipdata = f.read()
                except IOError:
                    raise AnsibleError('A different worker process failed to create module file. Look at traceback for that process for debugging information.')
        zipdata = to_text(zipdata, errors='surrogate_or_strict')
        (o_interpreter, o_args) = _extract_interpreter(b_module_data)
        if o_interpreter is None:
            o_interpreter = u'/usr/bin/python'
        (shebang, interpreter) = _get_shebang(o_interpreter, task_vars, templar, o_args, remote_is_local=remote_is_local)
        rlimit_nofile = C.config.get_config_value('PYTHON_MODULE_RLIMIT_NOFILE', variables=task_vars)
        if not isinstance(rlimit_nofile, int):
            rlimit_nofile = int(templar.template(rlimit_nofile))
        if rlimit_nofile:
            rlimit = ANSIBALLZ_RLIMIT_TEMPLATE % dict(rlimit_nofile=rlimit_nofile)
        else:
            rlimit = ''
        coverage_config = os.environ.get('_ANSIBLE_COVERAGE_CONFIG')
        if coverage_config:
            coverage_output = os.environ['_ANSIBLE_COVERAGE_OUTPUT']
            if coverage_output:
                coverage = ANSIBALLZ_COVERAGE_TEMPLATE % dict(coverage_config=coverage_config, coverage_output=coverage_output)
            else:
                coverage = ANSIBALLZ_COVERAGE_CHECK_TEMPLATE
        else:
            coverage = ''
        output.write(to_bytes(ACTIVE_ANSIBALLZ_TEMPLATE % dict(zipdata=zipdata, ansible_module=module_name, module_fqn=remote_module_fqn, params=python_repred_params, shebang=shebang, coding=ENCODING_STRING, date_time=date_time, coverage=coverage, rlimit=rlimit)))
        b_module_data = output.getvalue()
    elif module_substyle == 'powershell':
        shebang = u'#!powershell'
        b_module_data = ps_manifest._create_powershell_wrapper(b_module_data, module_path, module_args, environment, async_timeout, become, become_method, become_user, become_password, become_flags, module_substyle, task_vars, remote_module_fqn)
    elif module_substyle == 'jsonargs':
        module_args_json = to_bytes(json.dumps(module_args, cls=AnsibleJSONEncoder, vault_to_text=True))
        python_repred_args = to_bytes(repr(module_args_json))
        b_module_data = b_module_data.replace(REPLACER_VERSION, to_bytes(repr(__version__)))
        b_module_data = b_module_data.replace(REPLACER_COMPLEX, python_repred_args)
        b_module_data = b_module_data.replace(REPLACER_SELINUX, to_bytes(','.join(C.DEFAULT_SELINUX_SPECIAL_FS)))
        b_module_data = b_module_data.replace(REPLACER_JSONARGS, module_args_json)
        facility = b'syslog.' + to_bytes(task_vars.get('ansible_syslog_facility', C.DEFAULT_SYSLOG_FACILITY), errors='surrogate_or_strict')
        b_module_data = b_module_data.replace(b'syslog.LOG_USER', facility)
    return (b_module_data, module_style, shebang)

def _extract_interpreter(b_module_data):
    if False:
        i = 10
        return i + 15
    '\n    Used to extract shebang expression from binary module data and return a text\n    string with the shebang, or None if no shebang is detected.\n    '
    interpreter = None
    args = []
    b_lines = b_module_data.split(b'\n', 1)
    if b_lines[0].startswith(b'#!'):
        b_shebang = b_lines[0].strip()
        cli_split = shlex.split(to_text(b_shebang[2:], errors='surrogate_or_strict'))
        cli_split = [to_text(a, errors='surrogate_or_strict') for a in cli_split]
        interpreter = cli_split[0]
        args = cli_split[1:]
    return (interpreter, args)

def modify_module(module_name, module_path, module_args, templar, task_vars=None, module_compression='ZIP_STORED', async_timeout=0, become=False, become_method=None, become_user=None, become_password=None, become_flags=None, environment=None, remote_is_local=False):
    if False:
        print('Hello World!')
    "\n    Used to insert chunks of code into modules before transfer rather than\n    doing regular python imports.  This allows for more efficient transfer in\n    a non-bootstrapping scenario by not moving extra files over the wire and\n    also takes care of embedding arguments in the transferred modules.\n\n    This version is done in such a way that local imports can still be\n    used in the module code, so IDEs don't have to be aware of what is going on.\n\n    Example:\n\n    from ansible.module_utils.basic import *\n\n       ... will result in the insertion of basic.py into the module\n       from the module_utils/ directory in the source tree.\n\n    For powershell, this code effectively no-ops, as the exec wrapper requires access to a number of\n    properties not available here.\n\n    "
    task_vars = {} if task_vars is None else task_vars
    environment = {} if environment is None else environment
    with open(module_path, 'rb') as f:
        b_module_data = f.read()
    (b_module_data, module_style, shebang) = _find_module_utils(module_name, b_module_data, module_path, module_args, task_vars, templar, module_compression, async_timeout=async_timeout, become=become, become_method=become_method, become_user=become_user, become_password=become_password, become_flags=become_flags, environment=environment, remote_is_local=remote_is_local)
    if module_style == 'binary':
        return (b_module_data, module_style, to_text(shebang, nonstring='passthru'))
    elif shebang is None:
        (interpreter, args) = _extract_interpreter(b_module_data)
        if interpreter is not None:
            (shebang, new_interpreter) = _get_shebang(interpreter, task_vars, templar, args, remote_is_local=remote_is_local)
            b_lines = b_module_data.split(b'\n', 1)
            if interpreter != new_interpreter:
                b_lines[0] = to_bytes(shebang, errors='surrogate_or_strict', nonstring='passthru')
            if os.path.basename(interpreter).startswith(u'python'):
                b_lines.insert(1, b_ENCODING_STRING)
            b_module_data = b'\n'.join(b_lines)
    return (b_module_data, module_style, shebang)

def get_action_args_with_defaults(action, args, defaults, templar, action_groups=None):
    if False:
        return 10
    if action_groups is None:
        msg = 'Finding module_defaults for action %s. The caller has not passed the action_groups, so any that may include this action will be ignored.'
        display.warning(msg=msg)
        group_names = []
    else:
        group_names = action_groups.get(action, [])
    tmp_args = {}
    module_defaults = {}
    if isinstance(defaults, list):
        for default in defaults:
            module_defaults.update(default)
    module_defaults = templar.template(module_defaults)
    for default in module_defaults:
        if default.startswith('group/'):
            group_name = default.split('group/')[-1]
            if group_name in group_names:
                tmp_args.update((module_defaults.get('group/%s' % group_name) or {}).copy())
    tmp_args.update(module_defaults.get(action, {}).copy())
    tmp_args.update(args)
    return tmp_args