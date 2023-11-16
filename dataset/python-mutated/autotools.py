import inspect
import os
import os.path
import stat
import subprocess
from typing import List
import llnl.util.filesystem as fs
import llnl.util.tty as tty
import spack.build_environment
import spack.builder
import spack.package_base
from spack.directives import build_system, conflicts, depends_on
from spack.multimethod import when
from spack.operating_systems.mac_os import macos_version
from spack.util.executable import Executable
from spack.version import Version
from ._checks import BaseBuilder, apply_macos_rpath_fixups, ensure_build_dependencies_or_raise, execute_build_time_tests, execute_install_time_tests

class AutotoolsPackage(spack.package_base.PackageBase):
    """Specialized class for packages built using GNU Autotools."""
    build_system_class = 'AutotoolsPackage'
    legacy_buildsystem = 'autotools'
    build_system('autotools')
    with when('build_system=autotools'):
        depends_on('gnuconfig', type='build', when='target=ppc64le:')
        depends_on('gnuconfig', type='build', when='target=aarch64:')
        depends_on('gnuconfig', type='build', when='target=riscv64:')
        depends_on('gmake', type='build')
        conflicts('platform=windows')

    def flags_to_build_system_args(self, flags):
        if False:
            print('Hello World!')
        'Produces a list of all command line arguments to pass specified\n        compiler flags to configure.'
        setattr(self, 'configure_flag_args', [])
        for (flag, values) in flags.items():
            if values:
                var_name = 'LIBS' if flag == 'ldlibs' else flag.upper()
                values_str = '{0}={1}'.format(var_name, ' '.join(values))
                self.configure_flag_args.append(values_str)
        values = flags.get('fflags', None)
        if values:
            values_str = 'FCFLAGS={0}'.format(' '.join(values))
            self.configure_flag_args.append(values_str)

    def enable_or_disable(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.builder.enable_or_disable(*args, **kwargs)

    def with_or_without(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.builder.with_or_without(*args, **kwargs)

@spack.builder.builder('autotools')
class AutotoolsBuilder(BaseBuilder):
    """The autotools builder encodes the default way of installing software built
    with autotools. It has four phases that can be overridden, if need be:

        1. :py:meth:`~.AutotoolsBuilder.autoreconf`
        2. :py:meth:`~.AutotoolsBuilder.configure`
        3. :py:meth:`~.AutotoolsBuilder.build`
        4. :py:meth:`~.AutotoolsBuilder.install`

    They all have sensible defaults and for many packages the only thing necessary
    is to override the helper method
    :meth:`~spack.build_systems.autotools.AutotoolsBuilder.configure_args`.

    For a finer tuning you may also override:

        +-----------------------------------------------+--------------------+
        | **Method**                                    | **Purpose**        |
        +===============================================+====================+
        | :py:attr:`~.AutotoolsBuilder.build_targets`   | Specify ``make``   |
        |                                               | targets for the    |
        |                                               | build phase        |
        +-----------------------------------------------+--------------------+
        | :py:attr:`~.AutotoolsBuilder.install_targets` | Specify ``make``   |
        |                                               | targets for the    |
        |                                               | install phase      |
        +-----------------------------------------------+--------------------+
        | :py:meth:`~.AutotoolsBuilder.check`           | Run  build time    |
        |                                               | tests if required  |
        +-----------------------------------------------+--------------------+

    """
    phases = ('autoreconf', 'configure', 'build', 'install')
    legacy_methods = ('configure_args', 'check', 'installcheck')
    legacy_attributes = ('archive_files', 'patch_libtool', 'build_targets', 'install_targets', 'build_time_test_callbacks', 'install_time_test_callbacks', 'force_autoreconf', 'autoreconf_extra_args', 'install_libtool_archives', 'patch_config_files', 'configure_directory', 'configure_abs_path', 'build_directory', 'autoreconf_search_path_args')
    patch_libtool = True
    build_targets: List[str] = []
    install_targets = ['install']
    build_time_test_callbacks = ['check']
    install_time_test_callbacks = ['installcheck']
    force_autoreconf = False
    autoreconf_extra_args: List[str] = []
    install_libtool_archives = False

    @property
    def patch_config_files(self):
        if False:
            return 10
        'Whether to update old ``config.guess`` and ``config.sub`` files\n        distributed with the tarball.\n\n        This currently only applies to ``ppc64le:``, ``aarch64:``, and\n        ``riscv64`` target architectures.\n\n        The substitutes are taken from the ``gnuconfig`` package, which is\n        automatically added as a build dependency for these architectures. In case\n        system versions of these config files are required, the ``gnuconfig`` package\n        can be marked external, with a prefix pointing to the directory containing the\n        system ``config.guess`` and ``config.sub`` files.\n        '
        return self.pkg.spec.satisfies('target=ppc64le:') or self.pkg.spec.satisfies('target=aarch64:') or self.pkg.spec.satisfies('target=riscv64:')

    @property
    def _removed_la_files_log(self):
        if False:
            print('Hello World!')
        'File containing the list of removed libtool archives'
        build_dir = self.build_directory
        if not os.path.isabs(self.build_directory):
            build_dir = os.path.join(self.pkg.stage.path, build_dir)
        return os.path.join(build_dir, 'removed_la_files.txt')

    @property
    def archive_files(self):
        if False:
            while True:
                i = 10
        'Files to archive for packages based on autotools'
        files = [os.path.join(self.build_directory, 'config.log')]
        if not self.install_libtool_archives:
            files.append(self._removed_la_files_log)
        return files

    @spack.builder.run_after('autoreconf')
    def _do_patch_config_files(self):
        if False:
            for i in range(10):
                print('nop')
        'Some packages ship with older config.guess/config.sub files and need to\n        have these updated when installed on a newer architecture.\n\n        In particular, config.guess fails for PPC64LE for version prior to a\n        2013-06-10 build date (automake 1.13.4) and for AArch64 and RISC-V.\n        '
        if not self.patch_config_files:
            return
        if self.pkg.spec.satisfies('target=ppc64le:'):
            config_arch = 'ppc64le'
        elif self.pkg.spec.satisfies('target=aarch64:'):
            config_arch = 'aarch64'
        elif self.pkg.spec.satisfies('target=riscv64:'):
            config_arch = 'riscv64'
        else:
            config_arch = 'local'

        def runs_ok(script_abs_path):
            if False:
                while True:
                    i = 10
            additional_args = {'config.sub': [config_arch]}
            script_name = os.path.basename(script_abs_path)
            args = [script_abs_path] + additional_args.get(script_name, [])
            try:
                subprocess.check_call(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                tty.debug(e)
                return False
            return True
        to_be_patched = fs.find(self.pkg.stage.path, files=['config.sub', 'config.guess'])
        to_be_patched = [f for f in to_be_patched if not runs_ok(f)]
        if not to_be_patched:
            return
        ensure_build_dependencies_or_raise(spec=self.pkg.spec, dependencies=['gnuconfig'], error_msg='Cannot patch config files')
        to_be_found = list(set((os.path.basename(f) for f in to_be_patched)))
        gnuconfig = self.pkg.spec['gnuconfig']
        gnuconfig_dir = gnuconfig.prefix
        if gnuconfig_dir is None:
            raise spack.build_environment.InstallError('Spack could not find substitutes for GNU config files because no prefix is available for the `gnuconfig` package. Make sure you set a prefix path instead of modules for external `gnuconfig`.')
        candidates = fs.find(gnuconfig_dir, files=to_be_found, recursive=False)
        if not candidates:
            msg = 'Spack could not find `config.guess` and `config.sub` files in the `gnuconfig` prefix `{0}`. This means the `gnuconfig` package is broken'.format(gnuconfig_dir)
            if gnuconfig.external:
                msg += ' or the `gnuconfig` package prefix is misconfigured as an external package'
            raise spack.build_environment.InstallError(msg)
        candidates = [f for f in candidates if runs_ok(f)]
        substitutes = {}
        for candidate in candidates:
            config_file = os.path.basename(candidate)
            substitutes[config_file] = candidate
            to_be_found.remove(config_file)
        if to_be_found:
            msg = 'Spack could not find working replacements for the following autotools config\nfiles: {0}.\n\nTo resolve this problem, please try the following:\n1. Try to rebuild with `patch_config_files = False` in the package `{1}`, to\n   rule out that Spack tries to replace config files not used by the build.\n2. Verify that the `gnuconfig` package is up-to-date.\n3. On some systems you need to use system-provided `config.guess` and `config.sub`\n   files. In this case, mark `gnuconfig` as an non-buildable external package,\n   and set the prefix to the directory containing the `config.guess` and\n   `config.sub` files.\n'
            raise spack.build_environment.InstallError(msg.format(', '.join(to_be_found), self.name))
        for abs_path in to_be_patched:
            name = os.path.basename(abs_path)
            mode = os.stat(abs_path).st_mode
            os.chmod(abs_path, stat.S_IWUSR)
            fs.copy(substitutes[name], abs_path)
            os.chmod(abs_path, mode)

    @spack.builder.run_before('configure')
    def _patch_usr_bin_file(self):
        if False:
            i = 10
            return i + 15
        'On NixOS file is not available in /usr/bin/file. Patch configure\n        scripts to use file from path.'
        if self.spec.os.startswith('nixos'):
            x = fs.FileFilter(*filter(fs.is_exe, fs.find(self.build_directory, 'configure', recursive=True)))
            with fs.keep_modification_time(*x.filenames):
                x.filter(regex='/usr/bin/file', repl='file', string=True)

    @spack.builder.run_before('configure')
    def _set_autotools_environment_variables(self):
        if False:
            i = 10
            return i + 15
        "Many autotools builds use a version of mknod.m4 that fails when\n        running as root unless FORCE_UNSAFE_CONFIGURE is set to 1.\n\n        We set this to 1 and expect the user to take responsibility if\n        they are running as root. They have to anyway, as this variable\n        doesn't actually prevent configure from doing bad things as root.\n        Without it, configure just fails halfway through, but it can\n        still run things *before* this check. Forcing this just removes a\n        nuisance -- this is not circumventing any real protection.\n        "
        os.environ['FORCE_UNSAFE_CONFIGURE'] = '1'

    @spack.builder.run_before('configure')
    def _do_patch_libtool_configure(self):
        if False:
            while True:
                i = 10
        'Patch bugs that propagate from libtool macros into "configure" and\n        further into "libtool". Note that patches that can be fixed by patching\n        "libtool" directly should be implemented in the _do_patch_libtool method\n        below.'
        if not self.patch_libtool:
            return
        x = fs.FileFilter(*filter(fs.is_exe, fs.find(self.build_directory, 'configure', recursive=True)))
        with fs.keep_modification_time(*x.filenames):
            x.filter(regex='^(\\s*if test x-L = )("\\$p" \\|\\|\\s*)$', repl='\\1x\\2')
            x.filter(regex='^(\\s*test x-R = )("\\$p")(; then\\s*)$', repl='\\1x\\2 || test x-l = x"$p"\\3')
            x.filter(regex='^(\\s*test \\$p = "-R")(; then\\s*)$', repl='\\1 || test x-l = x"$p"\\2')

    @spack.builder.run_after('configure')
    def _do_patch_libtool(self):
        if False:
            print('Hello World!')
        'If configure generates a "libtool" script that does not correctly\n        detect the compiler (and patch_libtool is set), patch in the correct\n        values for libtool variables.\n\n        The generated libtool script supports mixed compilers through tags:\n        ``libtool --tag=CC/CXX/FC/...```. For each tag there is a block with variables,\n        which defines what flags to pass to the compiler. The default variables (which\n        are used by the default tag CC) are set in a block enclosed by\n        ``# ### {BEGIN,END} LIBTOOL CONFIG``. For non-default tags, there are\n        corresponding blocks ``# ### {BEGIN,END} LIBTOOL TAG CONFIG: {CXX,FC,F77}`` at\n        the end of the file (after the exit command). libtool evals these blocks.\n        Whenever we need to update variables that the configure script got wrong\n        (for example cause it did not recognize the compiler), we should properly scope\n        those changes to these tags/blocks so they only apply to the compiler we care\n        about. Below, ``start_at`` and ``stop_at`` are used for that.'
        if not self.patch_libtool:
            return
        x = fs.FileFilter(*filter(fs.is_exe, fs.find(self.build_directory, 'libtool', recursive=True)))
        if not x.filenames:
            return
        markers = {'cc': 'LIBTOOL CONFIG'}
        for tag in ['cxx', 'fc', 'f77']:
            markers[tag] = 'LIBTOOL TAG CONFIG: {0}'.format(tag.upper())
        if self.pkg.compiler.name == 'nag':
            for tag in ['fc', 'f77']:
                marker = markers[tag]
                x.filter(regex='^wl=""$', repl='wl="{0}"'.format(self.pkg.compiler.linker_arg), start_at='# ### BEGIN {0}'.format(marker), stop_at='# ### END {0}'.format(marker))
        else:
            x.filter(regex='^wl=""$', repl='wl="{0}"'.format(self.pkg.compiler.linker_arg))
        for (cc, marker) in markers.items():
            x.filter(regex='^pic_flag=""$', repl='pic_flag="{0}"'.format(getattr(self.pkg.compiler, '{0}_pic_flag'.format(cc))), start_at='# ### BEGIN {0}'.format(marker), stop_at='# ### END {0}'.format(marker))
        if self.pkg.compiler.name == 'fj':
            x.filter(regex='-nostdlib', repl='', string=True)
            rehead = '/\\S*/'
            for o in ['fjhpctag\\.o', 'fjcrt0\\.o', 'fjlang08\\.o', 'fjomp\\.o', 'crti\\.o', 'crtbeginS\\.o', 'crtendS\\.o']:
                x.filter(regex=rehead + o, repl='')
        elif self.pkg.compiler.name == 'dpcpp':
            x.filter(regex='^(predep_objects=.*)/tmp/conftest-[0-9A-Fa-f]+\\.o', repl='\\1')
            x.filter(regex='^(predep_objects=.*)/tmp/a-[0-9A-Fa-f]+\\.o', repl='\\1')
        elif self.pkg.compiler.name == 'nag':
            for tag in ['fc', 'f77']:
                marker = markers[tag]
                start_at = '# ### BEGIN {0}'.format(marker)
                stop_at = '# ### END {0}'.format(marker)
                x.filter(regex='\\$CC -shared', repl='\\$CC -Wl,-shared', string=True, start_at=start_at, stop_at=stop_at)
                x.filter(regex='^whole_archive_flag_spec="\\\\\\$({?wl}?)--whole-archive\\\\\\$convenience \\\\\\$\\1--no-whole-archive"$', repl='whole_archive_flag_spec="\\$\\1--whole-archive\\`for conv in \\$convenience\\\\\\\\\\"\\\\\\\\\\"; do test -n \\\\\\\\\\"\\$conv\\\\\\\\\\" && new_convenience=\\\\\\\\\\"\\$new_convenience,\\$conv\\\\\\\\\\"; done; func_echo_all \\\\\\\\\\"\\$new_convenience\\\\\\\\\\"\\` \\$\\1--no-whole-archive"', start_at=start_at, stop_at=stop_at)
                x.filter(regex='^(with_gcc=.*)$', repl='\\1\n\n# Is the compiler the NAG compiler?\nwith_nag=yes', start_at=start_at, stop_at=stop_at)
            for tag in ['cc', 'cxx']:
                marker = markers[tag]
                x.filter(regex='^(with_gcc=.*)$', repl='\\1\n\n# Is the compiler the NAG compiler?\nwith_nag=no', start_at='# ### BEGIN {0}'.format(marker), stop_at='# ### END {0}'.format(marker))
            x.filter(regex='^(\\s*)(for tmp_inherited_linker_flag in \\$tmp_inherited_linker_flags; do\\s*)$', repl='\\1if test "x$with_nag" = xyes; then\n\\1  revert_nag_pthread=$tmp_inherited_linker_flags\n\\1  tmp_inherited_linker_flags=`$ECHO "$tmp_inherited_linker_flags" | $SED \'s% -pthread% -Wl,-pthread%g\'`\n\\1  test x"$revert_nag_pthread" = x"$tmp_inherited_linker_flags" && revert_nag_pthread=no || revert_nag_pthread=yes\n\\1fi\n\\1\\2', start_at='if test -n "$inherited_linker_flags"; then', stop_at='case " $new_inherited_linker_flags " in')
            start_at = '# Time to change all our "foo.ltframework" stuff back to "-framework foo"'
            stop_at = '# installed libraries to the beginning of the library search list'
            x.filter(regex='(\\s*)(# move library search paths that coincide with paths to not yet\\s*)$', repl='\\1test x"$with_nag$revert_nag_pthread" = xyesyes &&\n\\1  new_inherited_linker_flags=`$ECHO " $new_inherited_linker_flags" | $SED \'s% -Wl,-pthread% -pthread%g\'`\n\\1\\2', start_at=start_at, stop_at=stop_at)

    @property
    def configure_directory(self):
        if False:
            print('Hello World!')
        "Return the directory where 'configure' resides."
        return self.pkg.stage.source_path

    @property
    def configure_abs_path(self):
        if False:
            return 10
        configure_abs_path = os.path.join(os.path.abspath(self.configure_directory), 'configure')
        return configure_abs_path

    @property
    def build_directory(self):
        if False:
            print('Hello World!')
        'Override to provide another place to build the package'
        return self.configure_directory

    @spack.builder.run_before('autoreconf')
    def delete_configure_to_force_update(self):
        if False:
            return 10
        if self.force_autoreconf:
            fs.force_remove(self.configure_abs_path)

    def autoreconf(self, pkg, spec, prefix):
        if False:
            return 10
        'Not needed usually, configure should be already there'
        if os.path.exists(self.configure_abs_path):
            return
        ensure_build_dependencies_or_raise(spec=spec, dependencies=['autoconf', 'automake', 'libtool'], error_msg='Cannot generate configure')
        tty.msg('Configure script not found: trying to generate it')
        tty.warn('*********************************************************')
        tty.warn('* If the default procedure fails, consider implementing *')
        tty.warn('*        a custom AUTORECONF phase in the package       *')
        tty.warn('*********************************************************')
        with fs.working_dir(self.configure_directory):
            m = inspect.getmodule(self.pkg)
            autoreconf_args = ['-ivf']
            autoreconf_args += self.autoreconf_search_path_args
            autoreconf_args += self.autoreconf_extra_args
            m.autoreconf(*autoreconf_args)

    @property
    def autoreconf_search_path_args(self):
        if False:
            print('Hello World!')
        'Search path includes for autoreconf. Add an -I flag for all `aclocal` dirs\n        of build deps, skips the default path of automake, move external include\n        flags to the back, since they might pull in unrelated m4 files shadowing\n        spack dependencies.'
        return _autoreconf_search_path_args(self.spec)

    @spack.builder.run_after('autoreconf')
    def set_configure_or_die(self):
        if False:
            i = 10
            return i + 15
        'Ensure the presence of a "configure" script, or raise. If the "configure"\n        is found, a module level attribute is set.\n\n        Raises:\n             RuntimeError: if the "configure" script is not found\n        '
        if not os.path.exists(self.configure_abs_path):
            msg = 'configure script not found in {0}'
            raise RuntimeError(msg.format(self.configure_directory))
        inspect.getmodule(self.pkg).configure = Executable(self.configure_abs_path)

    def configure_args(self):
        if False:
            while True:
                i = 10
        'Return the list of all the arguments that must be passed to configure,\n        except ``--prefix`` which will be pre-pended to the list.\n        '
        return []

    def configure(self, pkg, spec, prefix):
        if False:
            while True:
                i = 10
        'Run "configure", with the arguments specified by the builder and an\n        appropriately set prefix.\n        '
        options = getattr(self.pkg, 'configure_flag_args', [])
        options += ['--prefix={0}'.format(prefix)]
        options += self.configure_args()
        with fs.working_dir(self.build_directory, create=True):
            inspect.getmodule(self.pkg).configure(*options)

    def build(self, pkg, spec, prefix):
        if False:
            while True:
                i = 10
        'Run "make" on the build targets specified by the builder.'
        params = ['V=1']
        params += self.build_targets
        with fs.working_dir(self.build_directory):
            inspect.getmodule(self.pkg).make(*params)

    def install(self, pkg, spec, prefix):
        if False:
            while True:
                i = 10
        'Run "make" on the install targets specified by the builder.'
        with fs.working_dir(self.build_directory):
            inspect.getmodule(self.pkg).make(*self.install_targets)
    spack.builder.run_after('build')(execute_build_time_tests)

    def check(self):
        if False:
            while True:
                i = 10
        'Run "make" on the ``test`` and ``check`` targets, if found.'
        with fs.working_dir(self.build_directory):
            self.pkg._if_make_target_execute('test')
            self.pkg._if_make_target_execute('check')

    def _activate_or_not(self, name, activation_word, deactivation_word, activation_value=None, variant=None):
        if False:
            print('Hello World!')
        "This function contain the current implementation details of\n        :meth:`~spack.build_systems.autotools.AutotoolsBuilder.with_or_without` and\n        :meth:`~spack.build_systems.autotools.AutotoolsBuilder.enable_or_disable`.\n\n        Args:\n            name (str): name of the option that is being activated or not\n            activation_word (str): the default activation word ('with' in the\n                case of ``with_or_without``)\n            deactivation_word (str): the default deactivation word ('without'\n                in the case of ``with_or_without``)\n            activation_value (typing.Callable): callable that accepts a single\n                value. This value is either one of the allowed values for a\n                multi-valued variant or the name of a bool-valued variant.\n                Returns the parameter to be used when the value is activated.\n\n                The special value 'prefix' can also be assigned and will return\n                ``spec[name].prefix`` as activation parameter.\n            variant (str): name of the variant that is being processed\n                           (if different from option name)\n\n        Examples:\n\n            Given a package with:\n\n            .. code-block:: python\n\n                variant('foo', values=('x', 'y'), description='')\n                variant('bar', default=True, description='')\n                variant('ba_z', default=True, description='')\n\n            calling this function like:\n\n            .. code-block:: python\n\n                _activate_or_not(\n                    'foo', 'with', 'without', activation_value='prefix'\n                )\n                _activate_or_not('bar', 'with', 'without')\n                _activate_or_not('ba-z', 'with', 'without', variant='ba_z')\n\n            will generate the following configuration options:\n\n            .. code-block:: console\n\n                --with-x=<prefix-to-x> --without-y --with-bar --with-ba-z\n\n            for ``<spec-name> foo=x +bar``\n\n        Note: returns an empty list when the variant is conditional and its condition\n              is not met.\n\n        Returns:\n            list: list of strings that corresponds to the activation/deactivation\n            of the variant that has been processed\n\n        Raises:\n            KeyError: if name is not among known variants\n        "
        spec = self.pkg.spec
        args = []
        if activation_value == 'prefix':
            activation_value = lambda x: spec[x].prefix
        variant = variant or name
        if variant not in self.pkg.variants:
            msg = '"{0}" is not a variant of "{1}"'
            raise KeyError(msg.format(variant, self.pkg.name))
        if variant not in spec.variants:
            return []
        (variant_desc, _) = self.pkg.variants[variant]
        if set(variant_desc.values) == set((True, False)):
            condition = '+{name}'.format(name=variant)
            options = [(name, condition in spec)]
        else:
            condition = '{variant}={value}'
            feature_values = getattr(variant_desc.values, 'feature_values', None) or variant_desc.values
            options = [(value, condition.format(variant=variant, value=value) in spec) for value in feature_values]
        for (option_value, activated) in options:
            override_name = '{0}_or_{1}_{2}'.format(activation_word, deactivation_word, option_value)
            line_generator = getattr(self, override_name, None) or getattr(self.pkg, override_name, None)
            if line_generator is None:

                def _default_generator(is_activated):
                    if False:
                        i = 10
                        return i + 15
                    if is_activated:
                        line = '--{0}-{1}'.format(activation_word, option_value)
                        if activation_value is not None and activation_value(option_value):
                            line += '={0}'.format(activation_value(option_value))
                        return line
                    return '--{0}-{1}'.format(deactivation_word, option_value)
                line_generator = _default_generator
            args.append(line_generator(activated))
        return args

    def with_or_without(self, name, activation_value=None, variant=None):
        if False:
            for i in range(10):
                print('nop')
        "Inspects a variant and returns the arguments that activate\n        or deactivate the selected feature(s) for the configure options.\n\n        This function works on all type of variants. For bool-valued variants\n        it will return by default ``--with-{name}`` or ``--without-{name}``.\n        For other kinds of variants it will cycle over the allowed values and\n        return either ``--with-{value}`` or ``--without-{value}``.\n\n        If activation_value is given, then for each possible value of the\n        variant, the option ``--with-{value}=activation_value(value)`` or\n        ``--without-{value}`` will be added depending on whether or not\n        ``variant=value`` is in the spec.\n\n        Args:\n            name (str): name of a valid multi-valued variant\n            activation_value (typing.Callable): callable that accepts a single\n                value and returns the parameter to be used leading to an entry\n                of the type ``--with-{name}={parameter}``.\n\n                The special value 'prefix' can also be assigned and will return\n                ``spec[name].prefix`` as activation parameter.\n\n        Returns:\n            list of arguments to configure\n        "
        return self._activate_or_not(name, 'with', 'without', activation_value, variant)

    def enable_or_disable(self, name, activation_value=None, variant=None):
        if False:
            for i in range(10):
                print('nop')
        "Same as\n        :meth:`~spack.build_systems.autotools.AutotoolsBuilder.with_or_without`\n        but substitute ``with`` with ``enable`` and ``without`` with ``disable``.\n\n        Args:\n            name (str): name of a valid multi-valued variant\n            activation_value (typing.Callable): if present accepts a single value\n                and returns the parameter to be used leading to an entry of the\n                type ``--enable-{name}={parameter}``\n\n                The special value 'prefix' can also be assigned and will return\n                ``spec[name].prefix`` as activation parameter.\n\n        Returns:\n            list of arguments to configure\n        "
        return self._activate_or_not(name, 'enable', 'disable', activation_value, variant)
    spack.builder.run_after('install')(execute_install_time_tests)

    def installcheck(self):
        if False:
            for i in range(10):
                print('nop')
        'Run "make" on the ``installcheck`` target, if found.'
        with fs.working_dir(self.build_directory):
            self.pkg._if_make_target_execute('installcheck')

    @spack.builder.run_after('install')
    def remove_libtool_archives(self):
        if False:
            return 10
        'Remove all .la files in prefix sub-folders if the package sets\n        ``install_libtool_archives`` to be False.\n        '
        if self.install_libtool_archives:
            return
        libtool_files = fs.find(str(self.pkg.prefix), '*.la', recursive=True)
        with fs.safe_remove(*libtool_files):
            fs.mkdirp(os.path.dirname(self._removed_la_files_log))
            with open(self._removed_la_files_log, mode='w') as f:
                f.write('\n'.join(libtool_files))

    def setup_build_environment(self, env):
        if False:
            print('Hello World!')
        if self.spec.platform == 'darwin' and macos_version() >= Version('11'):
            env.set('MACOSX_DEPLOYMENT_TARGET', '10.16')
    spack.builder.run_after('install', when='platform=darwin')(apply_macos_rpath_fixups)

def _autoreconf_search_path_args(spec):
    if False:
        return 10
    dirs_seen = set()
    (flags_spack, flags_external) = ([], [])
    for automake in spec.dependencies(name='automake', deptype='build'):
        try:
            s = os.stat(automake.prefix.share.aclocal)
            if stat.S_ISDIR(s.st_mode):
                dirs_seen.add((s.st_ino, s.st_dev))
        except OSError:
            pass
    for dep in spec.dependencies(deptype='build'):
        path = dep.prefix.share.aclocal
        try:
            s = os.stat(path)
        except OSError:
            continue
        if (s.st_ino, s.st_dev) in dirs_seen or not stat.S_ISDIR(s.st_mode):
            continue
        dirs_seen.add((s.st_ino, s.st_dev))
        flags = flags_external if dep.external else flags_spack
        flags.extend(['-I', path])
    return flags_spack + flags_external