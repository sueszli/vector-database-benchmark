""" Command line options of Nuitka.

These provide only the optparse options to use, and the mechanic to actually
do it, but updating and checking module "nuitka.Options" values is not in
the scope, to make sure it can be used without.

Note: This is using "optparse", because "argparse" is only Python 2.7 and
higher, and we still support Python 2.6 due to the RHELs still being used,
and despite the long deprecation, it's in every later release, and actually
pretty good.
"""
import os
import re
import sys
from nuitka.PythonFlavors import getPythonFlavorName
from nuitka.utils.CommandLineOptions import SUPPRESS_HELP, makeOptionsParser
from nuitka.utils.FileOperations import getFileContentByLine
from nuitka.utils.Utils import getArchitecture, getLinuxDistribution, getOS, getWindowsRelease, isLinux, isMacOS, isWin32OrPosixWindows, isWin32Windows, withNoSyntaxWarning
from nuitka.Version import getCommercialVersion, getNuitkaVersion
is_nuitka_run = os.path.basename(sys.argv[0]).lower().endswith('-run')
if not is_nuitka_run:
    usage = 'usage: %prog [--module] [--run] [options] main_module.py'
else:
    usage = 'usage: %prog [options] main_module.py'
parser = makeOptionsParser(usage=usage)
parser.add_option('--version', dest='version', action='store_true', default=False, require_compiling=False, help='Show version information and important details for bug reports, then exit. Defaults to off.')
parser.add_option('--module', action='store_true', dest='module_mode', default=False, help='Create an extension module executable instead of a program. Defaults to off.')
parser.add_option('--standalone', action='store_true', dest='is_standalone', default=False, help='Enable standalone mode for output. This allows you to transfer the created binary\nto other machines without it using an existing Python installation. This also\nmeans it will become big. It implies these option: "--follow-imports" and\n"--python-flag=no_site". Defaults to off.')
parser.add_option('--no-standalone', action='store_false', dest='is_standalone', default=False, help=SUPPRESS_HELP)
parser.add_option('--onefile', action='store_true', dest='is_onefile', default=False, help='On top of standalone mode, enable onefile mode. This means not a folder,\nbut a compressed executable is created and used. Defaults to off.')
parser.add_option('--no-onefile', action='store_false', dest='is_onefile', default=False, help=SUPPRESS_HELP)
parser.add_option('--python-debug', action='store_true', dest='python_debug', default=None, help='Use debug version or not. Default uses what you are using to run Nuitka, most\nlikely a non-debug version.')
parser.add_option('--python-flag', action='append', dest='python_flags', metavar='FLAG', default=[], help='Python flags to use. Default is what you are using to run Nuitka, this\nenforces a specific mode. These are options that also exist to standard\nPython executable. Currently supported: "-S" (alias "no_site"),\n"static_hashes" (do not use hash randomization), "no_warnings" (do not\ngive Python run time warnings), "-O" (alias "no_asserts"), "no_docstrings"\n(do not use doc strings), "-u" (alias "unbuffered"), "isolated" (do not\nload outside code) and "-m" (package mode, compile as "package.__main__").\nDefault empty.')
parser.add_option('--python-for-scons', action='store', dest='python_scons', metavar='PATH', default=None, help='If using Python3.3 or Python3.4, provide the path of a Python binary to use\nfor Scons. Otherwise Nuitka can use what you run Nuitka with or a Python\ninstallation from Windows registry. On Windows Python 3.5 or higher is\nneeded. On non-Windows, Python 2.6 or 2.7 will do as well.')
parser.add_option('--main', '--script-name', action='append', dest='mains', metavar='PATH', default=[], help='If specified once, this takes the place of the\npositional argument, i.e. the filename to compile.\nWhen given multiple times, it enables "multidist"\n(see User Manual) it allows you to create binaries\nthat depending on file name or invocation name.\n')
parser.add_option('--github-workflow-options', action='store_true', dest='github_workflow_options', default=False, help=SUPPRESS_HELP)
include_group = parser.add_option_group('Control the inclusion of modules and packages in result')
include_group.add_option('--include-package', action='append', dest='include_packages', metavar='PACKAGE', default=[], help='Include a whole package. Give as a Python namespace, e.g. "some_package.sub_package"\nand Nuitka will then find it and include it and all the modules found below that\ndisk location in the binary or extension module it creates, and make it available\nfor import by the code. To avoid unwanted sub packages, e.g. tests you can e.g. do\nthis "--nofollow-import-to=*.tests". Default empty.')
include_group.add_option('--include-module', action='append', dest='include_modules', metavar='MODULE', default=[], help='Include a single module. Give as a Python namespace, e.g. "some_package.some_module"\nand Nuitka will then find it and include it in the binary or extension module\nit creates, and make it available for import by the code. Default empty.')
include_group.add_option('--include-plugin-directory', action='append', dest='include_extra', metavar='MODULE/PACKAGE', default=[], help='Include also the code found in that directory, considering as if\nthey are each given as a main file. Overrides all other inclusion\noptions. You ought to prefer other inclusion options, that go by\nnames, rather than filenames, those find things through being in\n"sys.path". This option is for very special use cases only. Can\nbe given multiple times. Default empty.')
include_group.add_option('--include-plugin-files', action='append', dest='include_extra_files', metavar='PATTERN', default=[], help='Include into files matching the PATTERN. Overrides all other follow options.\nCan be given multiple times. Default empty.')
include_group.add_option('--prefer-source-code', action='store_true', dest='prefer_source_code', default=None, help='For already compiled extension modules, where there is both a source file and an\nextension module, normally the extension module is used, but it should be better\nto compile the module from available source code for best performance. If not\ndesired, there is --no-prefer-source-code to disable warnings about it. Default\noff.')
include_group.add_option('--no-prefer-source-code', action='store_false', dest='prefer_source_code', default=None, help=SUPPRESS_HELP)
del include_group
follow_group = parser.add_option_group('Control the following into imported modules')
follow_group.add_option('--follow-imports', action='store_true', dest='follow_all', default=None, help='Descend into all imported modules. Defaults to on in standalone mode, otherwise off.')
follow_group.add_option('--follow-import-to', action='append', dest='follow_modules', metavar='MODULE/PACKAGE', default=[], help='Follow to that module if used, or if a package, to the whole package. Can be given\nmultiple times. Default empty.')
follow_group.add_option('--nofollow-import-to', action='append', dest='follow_not_modules', metavar='MODULE/PACKAGE', default=[], help='Do not follow to that module name even if used, or if a package name, to the\nwhole package in any case, overrides all other options. Can be given multiple\ntimes. Default empty.')
follow_group.add_option('--nofollow-imports', action='store_false', dest='follow_all', default=None, help='Do not descend into any imported modules at all, overrides all other inclusion\noptions and not usable for standalone mode. Defaults to off.')
follow_group.add_option('--follow-stdlib', action='store_true', dest='follow_stdlib', default=False, help="Also descend into imported modules from standard library. This will increase\nthe compilation time by a lot and is also not well tested at this time and\nsometimes won't work. Defaults to off.")
del follow_group
onefile_group = parser.add_option_group('Onefile options')
onefile_group.add_option('--onefile-tempdir-spec', action='store', dest='onefile_tempdir_spec', metavar='ONEFILE_TEMPDIR_SPEC', default=None, help="Use this as a folder to unpack to in onefile mode. Defaults to\n'%TEMP%/onefile_%PID%_%TIME%', i.e. user temporary directory\nand being non-static it's removed. Use e.g. a string like\n'%CACHE_DIR%/%COMPANY%/%PRODUCT%/%VERSION%' which is a good\nstatic cache path, this will then not be removed.")
onefile_group.add_option('--onefile-child-grace-time', action='store', dest='onefile_child_grace_time', metavar='GRACE_TIME_MS', default=None, help='When stopping the child, e.g. due to CTRL-C or shutdown, etc. the\nPython code gets a "KeyboardInterrupt", that it may handle e.g. to\nflush data. This is the amount of time in ms, before the child it\nkilled in the hard way. Unit is ms, and default 5000.')
onefile_group.add_option('--onefile-no-compression', action='store_true', dest='onefile_no_compression', default=False, help='When creating the onefile, disable compression of the payload. This is\nmostly for debug purposes, or to save time. Default is off.')
onefile_group.add_option('--onefile-as-archive', action='store_true', dest='onefile_as_archive', default=False, help='When creating the onefile, use an archive format, that can be unpacked\nwith "nuitka-onefile-unpack" rather than a stream that only the onefile\nprogram itself unpacks. Default is off.')
del onefile_group
data_group = parser.add_option_group('Data files')
data_group.add_option('--include-package-data', action='append', dest='package_data', metavar='PACKAGE', default=[], help='Include data files for the given package name. DLLs and extension modules\nare not data files and never included like this. Can use patterns the\nfilenames as indicated below. Data files of packages are not included\nby default, but package configuration can do it.\nThis will only include non-DLL, non-extension modules, i.e. actual data\nfiles. After a ":" optionally a filename pattern can be given as\nwell, selecting only matching files. Examples:\n"--include-package-data=package_name" (all files)\n"--include-package-data=package_name=*.txt" (only certain type)\n"--include-package-data=package_name=some_filename.dat (concrete file)\nDefault empty.')
data_group.add_option('--include-data-files', action='append', dest='data_files', metavar='DESC', default=[], help="Include data files by filenames in the distribution. There are many\nallowed forms. With '--include-data-files=/path/to/file/*.txt=folder_name/some.txt' it\nwill copy a single file and complain if it's multiple. With\n'--include-data-files=/path/to/files/*.txt=folder_name/' it will put\nall matching files into that folder. For recursive copy there is a\nform with 3 values that '--include-data-files=/path/to/scan=folder_name=**/*.txt'\nthat will preserve directory structure. Default empty.")
data_group.add_option('--include-data-dir', action='append', dest='data_dirs', metavar='DIRECTORY', default=[], help="Include data files from complete directory in the distribution. This is\nrecursive. Check '--include-data-files' with patterns if you want non-recursive\ninclusion. An example would be '--include-data-dir=/path/some_dir=data/some_dir'\nfor plain copy, of the whole directory. All files are copied, if you want to\nexclude files you need to remove them beforehand, or use '--noinclude-data-files'\noption to remove them. Default empty.")
data_group.add_option('--noinclude-data-files', action='append', dest='data_files_inhibited', metavar='PATTERN', default=[], help="Do not include data files matching the filename pattern given. This is against\nthe target filename, not source paths. So to ignore a file pattern from package\ndata for 'package_name' should be matched as 'package_name/*.txt'. Or for the\nwhole directory simply use 'package_name'. Default empty.")
data_group.add_option('--list-package-data', action='store', dest='list_package_data', default='', require_compiling=False, help='Output the data files found for a given package name. Default not done.')
del data_group
metadata_group = parser.add_option_group('Metadata support')
metadata_group.add_option('--include-distribution-metadata', action='append', dest='include_distribution_metadata', metavar='DISTRIBUTION', default=[], help="Include metadata information for the given distribution name. Some packages\ncheck metadata for presence, version, entry points, etc. and without this\noption given, it only works when it's recognized at compile time which is\nnot always happening. This of course only makes sense for packages that are\nincluded in the compilation. Default empty.")
del metadata_group
dll_group = parser.add_option_group('DLL files')
dll_group.add_option('--noinclude-dlls', action='append', dest='dll_files_inhibited', metavar='PATTERN', default=[], help="Do not include DLL files matching the filename pattern given. This is against\nthe target filename, not source paths. So ignore a DLL 'someDLL' contained in\nthe package 'package_name' it should be matched as 'package_name/someDLL.*'.\nDefault empty.")
dll_group.add_option('--list-package-dlls', action='store', dest='list_package_dlls', default='', require_compiling=False, help='Output the DLLs found for a given package name. Default not done.')
del dll_group
warnings_group = parser.add_option_group('Control the warnings to be given by Nuitka')
warnings_group.add_option('--warn-implicit-exceptions', action='store_true', dest='warn_implicit_exceptions', default=False, help='Enable warnings for implicit exceptions detected at compile time.')
warnings_group.add_option('--warn-unusual-code', action='store_true', dest='warn_unusual_code', default=False, help='Enable warnings for unusual code detected at compile time.')
warnings_group.add_option('--assume-yes-for-downloads', action='store_true', dest='assume_yes_for_downloads', default=False, help='Allow Nuitka to download external code if necessary, e.g. dependency\nwalker, ccache, and even gcc on Windows. To disable, redirect input\nfrom nul device, e.g. "</dev/null" or "<NUL:". Default is to prompt.')
warnings_group.add_option('--nowarn-mnemonic', action='append', dest='nowarn_mnemonics', metavar='MNEMONIC', default=[], help='Disable warning for a given mnemonic. These are given to make sure you are aware of\ncertain topics, and typically point to the Nuitka website. The mnemonic is the part\nof the URL at the end, without the HTML suffix. Can be given multiple times and\naccepts shell pattern. Default empty.')
del warnings_group
execute_group = parser.add_option_group('Immediate execution after compilation')
execute_group.add_option('--run', action='store_true', dest='immediate_execution', default=is_nuitka_run, help='Execute immediately the created binary (or import the compiled module).\nDefaults to %s.' % ('on' if is_nuitka_run else 'off'))
execute_group.add_option('--debugger', '--gdb', action='store_true', dest='debugger', default=False, help='Execute inside a debugger, e.g. "gdb" or "lldb" to automatically get a stack trace.\nDefaults to off.')
execute_group.add_option('--execute-with-pythonpath', action='store_true', dest='keep_pythonpath', default=False, help="When immediately executing the created binary or module using '--run',\ndon't reset 'PYTHONPATH' environment. When all modules are successfully\nincluded, you ought to not need PYTHONPATH anymore, and definitely not\nfor standalone mode.")
del execute_group
compilation_group = parser.add_option_group('Compilation choices')
compilation_group.add_option('--user-package-configuration-file', action='append', dest='user_yaml_files', default=[], metavar='YAML_FILENAME', help='User provided Yaml file with package configuration. You can include DLLs,\nremove bloat, add hidden dependencies. Check User Manual for a complete\ndescription of the format to use. Can be given multiple times. Defaults\nto empty.')
compilation_group.add_option('--full-compat', action='store_false', dest='improved', default=True, help='Enforce absolute compatibility with CPython. Do not even allow minor\ndeviations from CPython behavior, e.g. not having better tracebacks\nor exception messages which are not really incompatible, but only\ndifferent or worse. This is intended for tests only and should *not*\nbe used.')
compilation_group.add_option('--file-reference-choice', action='store', dest='file_reference_mode', metavar='MODE', choices=('original', 'runtime', 'frozen'), default=None, help='Select what value "__file__" is going to be. With "runtime" (default for\nstandalone binary mode and module mode), the created binaries and modules,\nuse the location of themselves to deduct the value of "__file__". Included\npackages pretend to be in directories below that location. This allows you\nto include data files in deployments. If you merely seek acceleration, it\'s\nbetter for you to use the "original" value, where the source files location\nwill be used. With "frozen" a notation "<frozen module_name>" is used. For\ncompatibility reasons, the "__file__" value will always have ".py" suffix\nindependent of what it really is.')
compilation_group.add_option('--module-name-choice', action='store', dest='module_name_mode', metavar='MODE', choices=('original', 'runtime'), default=None, help='Select what value "__name__" and "__package__" are going to be. With "runtime"\n(default for module mode), the created module uses the parent package to\ndeduce the value of "__package__", to be fully compatible. The value "original"\n(default for other modes) allows for more static optimization to happen, but\nis incompatible for modules that normally can be loaded into any package.')
del compilation_group
output_group = parser.add_option_group('Output choices')
output_group.add_option('--output-filename', '-o', action='store', dest='output_filename', metavar='FILENAME', default=None, help="Specify how the executable should be named. For extension modules there is no\nchoice, also not for standalone mode and using it will be an error. This may\ninclude path information that needs to exist though. Defaults to '%s' on this\nplatform.\n" % '<program_name>' + ('.exe' if isWin32OrPosixWindows() else '.bin'))
output_group.add_option('--output-dir', action='store', dest='output_dir', metavar='DIRECTORY', default='', help='Specify where intermediate and final output files should be put. The DIRECTORY\nwill be populated with build folder, dist folder, binaries, etc.\nDefaults to current directory.\n')
output_group.add_option('--remove-output', action='store_true', dest='remove_build', default=False, help='Removes the build directory after producing the module or exe file.\nDefaults to off.')
output_group.add_option('--no-pyi-file', action='store_false', dest='pyi_file', default=True, help="Do not create a '.pyi' file for extension modules created by Nuitka. This is\nused to detect implicit imports.\nDefaults to off.")
del output_group
debug_group = parser.add_option_group('Debug features')
debug_group.add_option('--debug', action='store_true', dest='debug', default=False, help='Executing all self checks possible to find errors in Nuitka, do not use for\nproduction. Defaults to off.')
debug_group.add_option('--unstripped', '--unstriped', action='store_true', dest='unstripped', default=False, help='Keep debug info in the resulting object file for better debugger interaction.\nDefaults to off.')
debug_group.add_option('--profile', action='store_true', dest='profile', default=False, help='Enable vmprof based profiling of time spent. Not working currently. Defaults to off.')
debug_group.add_option('--internal-graph', action='store_true', dest='internal_graph', default=False, help='Create graph of optimization process internals, do not use for whole programs, but only\nfor small test cases. Defaults to off.')
debug_group.add_option('--trace-execution', action='store_true', dest='trace_execution', default=False, help='Traced execution output, output the line of code before executing it.\nDefaults to off.')
debug_group.add_option('--recompile-c-only', action='store_true', dest='recompile_c_only', default=False, help='This is not incremental compilation, but for Nuitka development only. Takes\nexisting files and simply compile them as C again. Allows compiling edited\nC files for quick debugging changes to the generated source, e.g. to see if\ncode is passed by, values output, etc, Defaults to off. Depends on compiling\nPython source to determine which files it should look at.')
debug_group.add_option('--xml', action='store', dest='xml_output', metavar='XML_FILENAME', default=None, help='Write the internal program structure, result of optimization in XML form to given filename.')
debug_group.add_option('--deployment', action='store_true', dest='is_deployment', default=False, help='Disable code aimed at making finding compatibility issues easier. This\nwill e.g. prevent execution with "-c" argument, which is often used by\ncode that attempts run a module, and causes a program to start itself\nover and over potentially. Default off.')
debug_group.add_option('--no-deployment-flag', action='append', dest='no_deployment_flags', metavar='FLAG', default=[], help='Keep deployment mode, but disable selectively parts of it. Errors from\ndeployment mode will output these identifiers. Default empty.')
debug_group.add_option('--experimental', action='append', dest='experimental', metavar='FLAG', default=[], help="Use features declared as 'experimental'. May have no effect if no experimental\nfeatures are present in the code. Uses secret tags (check source) per\nexperimented feature.")
debug_group.add_option('--explain-imports', action='store_true', dest='explain_imports', default=False, help=SUPPRESS_HELP)
debug_group.add_option('--low-memory', action='store_true', dest='low_memory', default=False, help='Attempt to use less memory, by forking less C compilation jobs and using\noptions that use less memory. For use on embedded machines. Use this in\ncase of out of memory problems. Defaults to off.')
debug_group.add_option('--create-environment-from-report', action='store', dest='create_environment_from_report', default='', require_compiling=False, help="Create a new virtualenv in that non-existing path from the report file given with\ne.g. '--report=compilation-report.xml'. Default not done.")
debug_group.add_option('--generate-c-only', action='store_true', dest='generate_c_only', default=False, help="Generate only C source code, and do not compile it to binary or module. This\nis for debugging and code coverage analysis that doesn't waste CPU. Defaults to\noff. Do not think you can use this directly.")
del debug_group
parser.add_option('--must-not-re-execute', action='store_false', dest='allow_reexecute', default=True, help=SUPPRESS_HELP)
parser.add_option('--edit-module-code', action='store', dest='edit_module_code', default=None, require_compiling=False, help=SUPPRESS_HELP)
c_compiler_group = parser.add_option_group('Backend C compiler choice')
c_compiler_group.add_option('--clang', action='store_true', dest='clang', default=False, help='Enforce the use of clang. On Windows this requires a working Visual\nStudio version to piggy back on. Defaults to off.')
c_compiler_group.add_option('--mingw64', action='store_true', dest='mingw64', default=False, help='Enforce the use of MinGW64 on Windows. Defaults to off unless MSYS2 with MinGW Python is used.')
c_compiler_group.add_option('--msvc', action='store', dest='msvc_version', default=None, help='Enforce the use of specific MSVC version on Windows. Allowed values\nare e.g. "14.3" (MSVC 2022) and other MSVC version numbers, specify\n"list" for a list of installed compilers, or use "latest".\n\nDefaults to latest MSVC being used if installed, otherwise MinGW64\nis used.')
c_compiler_group.add_option('--jobs', '-j', action='store', dest='jobs', metavar='N', default=None, help='Specify the allowed number of parallel C compiler jobs. Defaults to the\nsystem CPU count.')
c_compiler_group.add_option('--lto', action='store', dest='lto', metavar='choice', default='auto', choices=('yes', 'no', 'auto'), help='Use link time optimizations (MSVC, gcc, clang). Allowed values are\n"yes", "no", and "auto" (when it\'s known to work). Defaults to\n"auto".')
c_compiler_group.add_option('--static-libpython', action='store', dest='static_libpython', metavar='choice', default='auto', choices=('yes', 'no', 'auto'), help='Use static link library of Python. Allowed values are "yes", "no",\nand "auto" (when it\'s known to work). Defaults to "auto".')
del c_compiler_group
caching_group = parser.add_option_group('Cache Control')
_cache_names = ('all', 'ccache', 'bytecode', 'compression')
if isWin32Windows():
    _cache_names += ('dll-dependencies',)
caching_group.add_option('--disable-cache', action='append', dest='disabled_caches', choices=_cache_names, default=[], help='Disable selected caches, specify "all" for all cached. Currently allowed\nvalues are: %s. can be given multiple times or with comma separated values.\nDefault none.' % ','.join(('"%s"' % cache_name for cache_name in _cache_names)))
caching_group.add_option('--clean-cache', action='append', dest='clean_caches', choices=_cache_names, default=[], require_compiling=False, help='Clean the given caches before executing, specify "all" for all cached. Currently\nallowed values are: %s. can be given multiple times or with comma separated\nvalues. Default none.' % ','.join(('"%s"' % cache_name for cache_name in _cache_names)))
caching_group.add_option('--disable-bytecode-cache', action='store_true', dest='disable_bytecode_cache', default=False, help='Do not reuse dependency analysis results for modules, esp. from standard library,\nthat are included as bytecode. Same as --disable-cache=bytecode.')
caching_group.add_option('--disable-ccache', action='store_true', dest='disable_ccache', default=False, help='Do not attempt to use ccache (gcc, clang, etc.) or clcache (MSVC, clangcl).\nSame as --disable-cache=ccache.')
if isWin32Windows():
    caching_group.add_option('--disable-dll-dependency-cache', action='store_true', dest='disable_dll_dependency_cache', default=False, help='Disable the dependency walker cache. Will result in much longer times to create\nthe distribution folder, but might be used in case the cache is suspect to cause\nerrors. Same as --disable-cache=dll-dependencies.')
    caching_group.add_option('--force-dll-dependency-cache-update', action='store_true', dest='update_dependency_cache', default=False, help='For an update of the dependency walker cache. Will result in much longer times\nto create the distribution folder, but might be used in case the cache is suspect\nto cause errors or known to need an update.\n')
del caching_group
pgo_group = parser.add_option_group('PGO compilation choices')
pgo_group.add_option('--pgo', action='store_true', dest='is_c_pgo', default=False, help='Enables C level profile guided optimization (PGO), by executing a dedicated build first\nfor a profiling run, and then using the result to feedback into the C compilation.\nNote: This is experimental and not working with standalone modes of Nuitka yet.\nDefaults to off.')
pgo_group.add_option('--pgo-python', action='store_true', dest='is_python_pgo', default=False, help=SUPPRESS_HELP)
pgo_group.add_option('--pgo-python-input', action='store', dest='python_pgo_input', default=None, help=SUPPRESS_HELP)
pgo_group.add_option('--pgo-python-policy-unused-module', action='store', dest='python_pgo_policy_unused_module', choices=('include', 'exclude', 'bytecode'), default='include', help=SUPPRESS_HELP)
pgo_group.add_option('--pgo-args', action='store', dest='pgo_args', default='', help='Arguments to be passed in case of profile guided optimization. These are passed to the special\nbuilt executable during the PGO profiling run. Default empty.')
pgo_group.add_option('--pgo-executable', action='store', dest='pgo_executable', default=None, help='Command to execute when collecting profile information. Use this only, if you need to\nlaunch it through a script that prepares it to run. Default use created program.')
del pgo_group
tracing_group = parser.add_option_group('Tracing features')
tracing_group.add_option('--report', action='store', dest='compilation_report_filename', metavar='REPORT_FILENAME', default=None, help="Report module, data files, compilation, plugin, etc. details in an XML output file. This\nis also super useful for issue reporting. These reports can e.g. be used to re-create\nthe environment easily using it with '--create-environment-from-report', but contain a\nlot of information. Default is off.")
tracing_group.add_option('--report-diffable', action='store_true', dest='compilation_report_diffable', metavar='REPORT_DIFFABLE', default=False, help='Report data in diffable form, i.e. no timing or memory usage values that vary from run\nto run. Default is off.')
tracing_group.add_option('--report-user-provided', action='append', dest='compilation_report_user_data', metavar='KEY_VALUE', default=[], help="Report data from you. This can be given multiple times and be\nanything in 'key=value' form, where key should be an identifier, e.g. use\n'--report-user-provided=pipenv-lock-hash=64a5e4' to track some input values.\nDefault is empty.")
tracing_group.add_option('--report-template', action='append', dest='compilation_report_templates', metavar='REPORT_DESC', default=[], help="Report via template. Provide template and output filename 'template.rst.j2:output.rst'. For\nbuilt-in templates, check the User Manual for what these are. Can be given multiple times.\nDefault is empty.")
tracing_group.add_option('--quiet', action='store_true', dest='quiet', default=False, help='Disable all information outputs, but show warnings.\nDefaults to off.')
tracing_group.add_option('--show-scons', action='store_true', dest='show_scons', default=False, help='Run the C building backend Scons with verbose information, showing the executed commands,\ndetected compilers. Defaults to off.')
tracing_group.add_option('--no-progressbar', '--no-progress-bar', action='store_false', dest='progress_bar', default=True, help='Disable progress bars. Defaults to off.')
tracing_group.add_option('--show-progress', action='store_true', dest='show_progress', default=False, help='Obsolete: Provide progress information and statistics.\nDisables normal progress bar. Defaults to off.')
tracing_group.add_option('--show-memory', action='store_true', dest='show_memory', default=False, help='Provide memory information and statistics.\nDefaults to off.')
tracing_group.add_option('--show-modules', action='store_true', dest='show_inclusion', default=False, help="Provide information for included modules and DLLs\nObsolete: You should use '--report' file instead. Defaults to off.")
tracing_group.add_option('--show-modules-output', action='store', dest='show_inclusion_output', metavar='PATH', default=None, help="Where to output '--show-modules', should be a filename. Default is standard output.")
tracing_group.add_option('--verbose', action='store_true', dest='verbose', default=False, help='Output details of actions taken, esp. in optimizations. Can become a lot.\nDefaults to off.')
tracing_group.add_option('--verbose-output', action='store', dest='verbose_output', metavar='PATH', default=None, help="Where to output from '--verbose', should be a filename. Default is standard output.")
del tracing_group
os_group = parser.add_option_group('General OS controls')
os_group.add_option('--disable-console', '--macos-disable-console', '--windows-disable-console', action='store_true', dest='disable_console', default=None, help='When compiling for Windows or macOS, disable the console window and create a GUI\napplication. Defaults to off.')
os_group.add_option('--enable-console', action='store_false', dest='disable_console', default=None, help='When compiling for Windows or macOS, enable the console window and create a console\napplication. This disables hints from certain modules, e.g. "PySide" that suggest\nto disable it. Defaults to true.')
os_group.add_option('--force-stdout-spec', '--windows-force-stdout-spec', action='store', dest='force_stdout_spec', metavar='FORCE_STDOUT_SPEC', default=None, help="Force standard output of the program to go to this location. Useful for programs with\ndisabled console and programs using the Windows Services Plugin of Nuitka commercial.\nDefaults to not active, use e.g. '%PROGRAM_BASE%.out.txt', i.e. file near your program,\ncheck User Manual for full list of available values.")
os_group.add_option('--force-stderr-spec', '--windows-force-stderr-spec', action='store', dest='force_stderr_spec', metavar='FORCE_STDERR_SPEC', default=None, help="Force standard error of the program to go to this location. Useful for programs with\ndisabled console and programs using the Windows Services Plugin of Nuitka commercial.\nDefaults to not active, use e.g. '%PROGRAM_BASE%.err.txt', i.e. file near your program,\ncheck User Manual for full list of available values.")
del os_group
windows_group = parser.add_option_group('Windows specific controls')
windows_group.add_option('--windows-dependency-tool', action='store', dest='dependency_tool', default=None, help=SUPPRESS_HELP)
windows_group.add_option('--windows-icon-from-ico', action='append', dest='icon_path', metavar='ICON_PATH', default=[], help='Add executable icon. Can be given multiple times for different resolutions\nor files with multiple icons inside. In the later case, you may also suffix\nwith #<n> where n is an integer index starting from 1, specifying a specific\nicon to be included, and all others to be ignored.')
windows_group.add_option('--windows-icon-from-exe', action='store', dest='icon_exe_path', metavar='ICON_EXE_PATH', default=None, help='Copy executable icons from this existing executable (Windows only).')
windows_group.add_option('--onefile-windows-splash-screen-image', action='store', dest='splash_screen_image', default=None, help='When compiling for Windows and onefile, show this while loading the application. Defaults to off.')
windows_group.add_option('--windows-uac-admin', action='store_true', dest='windows_uac_admin', metavar='WINDOWS_UAC_ADMIN', default=False, help='Request Windows User Control, to grant admin rights on execution. (Windows only). Defaults to off.')
windows_group.add_option('--windows-uac-uiaccess', action='store_true', dest='windows_uac_uiaccess', metavar='WINDOWS_UAC_UIACCESS', default=False, help='Request Windows User Control, to enforce running from a few folders only, remote\ndesktop access. (Windows only). Defaults to off.')
del windows_group
macos_group = parser.add_option_group('macOS specific controls')
macos_group.add_option('--macos-target-arch', action='store', dest='macos_target_arch', choices=('universal', 'arm64', 'x86_64'), metavar='MACOS_TARGET_ARCH', default=None, help='What architectures is this to supposed to run on. Default and limit\nis what the running Python allows for. Default is "native" which is\nthe architecture the Python is run with.')
macos_group.add_option('--macos-create-app-bundle', action='store_true', dest='macos_create_bundle', default=False, help='When compiling for macOS, create a bundle rather than a plain binary\napplication. Currently experimental and incomplete. Currently this\nis the only way to unlock disabling of console.Defaults to off.')
macos_group.add_option('--macos-app-icon', action='append', dest='icon_path', metavar='ICON_PATH', default=[], help='Add icon for the application bundle to use. Can be given only one time. Defaults to Python icon if available.')
macos_group.add_option('--macos-signed-app-name', action='store', dest='macos_signed_app_name', metavar='MACOS_SIGNED_APP_NAME', default=None, help='Name of the application to use for macOS signing. Follow "com.YourCompany.AppName"\nnaming results for best results, as these have to be globally unique, and will\npotentially grant protected API accesses.')
macos_group.add_option('--macos-app-name', action='store', dest='macos_app_name', metavar='MACOS_APP_NAME', default=None, help='Name of the product to use in macOS bundle information. Defaults to base\nfilename of the binary.')
macos_group.add_option('--macos-app-mode', action='store', dest='macos_app_mode', metavar='MODE', choices=('gui', 'background', 'ui-element'), default='gui', help='Mode of application for the application bundle. When launching a Window, and appearing\nin Docker is desired, default value "gui" is a good fit. Without a Window ever, the\napplication is a "background" application. For UI elements that get to display later,\n"ui-element" is in-between. The application will not appear in dock, but get full\naccess to desktop when it does open a Window later.')
macos_group.add_option('--macos-sign-identity', action='store', dest='macos_sign_identity', metavar='MACOS_APP_VERSION', default='ad-hoc', help='When signing on macOS, by default an ad-hoc identify will be used, but with this\noption your get to specify another identity to use. The signing of code is now\nmandatory on macOS and cannot be disabled. Default "ad-hoc" if not given.')
macos_group.add_option('--macos-sign-notarization', action='store_true', dest='macos_sign_notarization', default=False, help='When signing for notarization, using a proper TeamID identity from Apple, use\nthe required runtime signing option, such that it can be accepted.')
macos_group.add_option('--macos-app-version', action='store', dest='macos_app_version', metavar='MACOS_APP_VERSION', default=None, help='Product version to use in macOS bundle information. Defaults to "1.0" if\nnot given.')
macos_group.add_option('--macos-app-protected-resource', action='append', dest='macos_protected_resources', metavar='RESOURCE_DESC', default=[], help='Request an entitlement for access to a macOS protected resources, e.g.\n"NSMicrophoneUsageDescription:Microphone access for recording audio."\nrequests access to the microphone and provides an informative text for\nthe user, why that is needed. Before the colon, is an OS identifier for\nan access right, then the informative text. Legal values can be found on\nhttps://developer.apple.com/documentation/bundleresources/information_property_list/protected_resources and\nthe option can be specified multiple times. Default empty.')
del macos_group
linux_group = parser.add_option_group('Linux specific controls')
linux_group.add_option('--linux-icon', '--linux-onefile-icon', action='append', dest='icon_path', metavar='ICON_PATH', default=[], help='Add executable icon for onefile binary to use. Can be given only one time. Defaults to Python icon if available.')
del linux_group
version_group = parser.add_option_group('Binary Version Information')
version_group.add_option('--company-name', '--windows-company-name', action='store', dest='company_name', metavar='COMPANY_NAME', default=None, help='Name of the company to use in version information. Defaults to unused.')
version_group.add_option('--product-name', '--windows-product-name', action='store', dest='product_name', metavar='PRODUCT_NAME', default=None, help='Name of the product to use in version information. Defaults to base filename of the binary.')
version_group.add_option('--file-version', '--windows-file-version', action='store', dest='file_version', metavar='FILE_VERSION', default=None, help='File version to use in version information. Must be a sequence of up to 4\nnumbers, e.g. 1.0 or 1.0.0.0, no more digits are allowed, no strings are\nallowed. Defaults to unused.')
version_group.add_option('--product-version', '--windows-product-version', action='store', dest='product_version', metavar='PRODUCT_VERSION', default=None, help='Product version to use in version information. Same rules as for file version.\nDefaults to unused.')
version_group.add_option('--file-description', '--windows-file-description', action='store', dest='file_description', metavar='FILE_DESCRIPTION', default=None, help='Description of the file used in version information. Windows only at this time. Defaults to binary filename.')
version_group.add_option('--copyright', action='store', dest='legal_copyright', metavar='COPYRIGHT_TEXT', default=None, help='Copyright used in version information. Windows only at this time. Defaults to not present.')
version_group.add_option('--trademarks', action='store', dest='legal_trademarks', metavar='TRADEMARK_TEXT', default=None, help='Trademark used in version information. Windows only at this time. Defaults to not present.')
del version_group
plugin_group = parser.add_option_group('Plugin control')
plugin_group.add_option('--enable-plugins', '--plugin-enable', action='append', dest='plugins_enabled', metavar='PLUGIN_NAME', default=[], help="Enabled plugins. Must be plug-in names. Use '--plugin-list' to query the\nfull list and exit. Default empty.")
plugin_group.add_option('--disable-plugins', '--plugin-disable', action='append', dest='plugins_disabled', metavar='PLUGIN_NAME', default=[], help="Disabled plugins. Must be plug-in names. Use '--plugin-list' to query the\nfull list and exit. Most standard plugins are not a good idea to disable.\nDefault empty.")
plugin_group.add_option('--plugin-no-detection', action='store_false', dest='detect_missing_plugins', default=True, help='Plugins can detect if they might be used, and the you can disable the warning\nvia "--disable-plugin=plugin-that-warned", or you can use this option to disable\nthe mechanism entirely, which also speeds up compilation slightly of course as\nthis detection code is run in vain once you are certain of which plugins to\nuse. Defaults to off.')
plugin_group.add_option('--plugin-list', action='store_true', dest='plugin_list', default=False, require_compiling=False, help='Show list of all available plugins and exit. Defaults to off.')
plugin_group.add_option('--user-plugin', action='append', dest='user_plugins', metavar='PATH', default=[], help='The file name of user plugin. Can be given multiple times. Default empty.')
plugin_group.add_option('--show-source-changes', action='store_true', dest='show_source_changes', default=False, help='Show source changes to original Python file content before compilation. Mostly\nintended for developing plugins. Default False.')
del plugin_group

def _considerPluginOptions(logger):
    if False:
        i = 10
        return i + 15
    from nuitka.plugins.Plugins import addPluginCommandLineOptions, addStandardPluginCommandLineOptions, addUserPluginCommandLineOptions
    addStandardPluginCommandLineOptions(parser=parser)
    for arg in sys.argv[1:]:
        if arg.startswith(('--enable-plugin=', '--enable-plugins=', '--plugin-enable=')):
            plugin_names = arg.split('=', 1)[1]
            if '=' in plugin_names:
                logger.sysexit("Error, plugin options format changed. Use '--enable-plugin=%s --help' to know new options." % plugin_names.split('=', 1)[0])
            addPluginCommandLineOptions(parser=parser, plugin_names=plugin_names.split(','))
        if arg.startswith('--user-plugin='):
            plugin_name = arg[14:]
            if '=' in plugin_name:
                logger.sysexit("Error, plugin options format changed. Use '--user-plugin=%s --help' to know new options." % plugin_name.split('=', 1)[0])
            addUserPluginCommandLineOptions(parser=parser, filename=plugin_name)

def _expandProjectArg(arg, filename_arg, for_eval):
    if False:
        return 10

    def wrap(value):
        if False:
            i = 10
            return i + 15
        if for_eval:
            return repr(value)
        else:
            return value
    values = {'OS': wrap(getOS()), 'Arch': wrap(getArchitecture()), 'Flavor': wrap(getPythonFlavorName()), 'Version': getNuitkaVersion(), 'Commercial': wrap(getCommercialVersion()), 'MAIN_DIRECTORY': wrap(os.path.dirname(filename_arg) or '.')}
    if isLinux():
        dist_info = getLinuxDistribution()
    else:
        dist_info = ('N/A', 'N/A', '0')
    values['Linux_Distribution_Name'] = dist_info[0]
    values['Linux_Distribution_Base'] = dist_info[1] or dist_info[0]
    values['Linux_Distribution_Version'] = dist_info[2]
    if isWin32OrPosixWindows():
        values['WindowsRelease'] = getWindowsRelease()
    arg = arg.format(**values)
    return arg

def getNuitkaProjectOptions(logger, filename_arg, module_mode):
    if False:
        return 10
    'Extract the Nuitka project options.\n\n    Note: This is used by Nuitka project and test tools as well.\n    '
    if os.path.isdir(filename_arg):
        if module_mode:
            filename_arg = os.path.join(filename_arg, '__init__.py')
        else:
            filename_arg = os.path.join(filename_arg, '__main__.py')
    try:
        contents_by_line = getFileContentByLine(filename_arg, 'rb')
    except (OSError, IOError):
        return

    def sysexit(count, message):
        if False:
            return 10
        logger.sysexit('%s:%d %s' % (filename_arg, count + 1, message))
    execute_block = True
    expect_block = False
    cond_level = -1
    for (line_number, line) in enumerate(contents_by_line):
        match = re.match(b'^\\s*#(\\s*)nuitka-project(.*?):(.*)', line)
        if match:
            (level, command, arg) = match.groups()
            level = len(level)
            arg = arg.rstrip()
            if expect_block and level <= cond_level:
                sysexit(line_number, "Error, 'nuitka-project-if|else' is expected to be followed by block start.")
            expect_block = False
            if level == cond_level and command == b'-else':
                execute_block = not execute_block
            elif level <= cond_level:
                execute_block = True
            if level > cond_level and (not execute_block):
                continue
            if str is not bytes:
                command = command.decode('utf8')
                arg = arg.decode('utf8')
            if command == '-if':
                if not arg.endswith(':'):
                    sysexit(line_number, "Error, 'nuitka-project-if' needs to start a block with a colon at line end.")
                arg = arg[:-1].strip()
                expanded = _expandProjectArg(arg, filename_arg, for_eval=True)
                with withNoSyntaxWarning():
                    r = eval(expanded)
                if r is not True and r is not False:
                    sys.exit("Error, 'nuitka-project-if' condition %r (expanded to %r) does not yield boolean result %r" % (arg, expanded, r))
                execute_block = r
                expect_block = True
                cond_level = level
            elif command == '-else':
                if arg:
                    sysexit(line_number, "Error, 'nuitka-project-else' cannot have argument.")
                if cond_level != level:
                    sysexit(line_number, "Error, 'nuitka-project-else' not currently allowed after nested nuitka-project-if.")
                expect_block = True
                cond_level = level
            elif command == '':
                arg = re.sub('^([\\w-]*=)([\'"])(.*)\\2$', '\\1\\3', arg.lstrip())
                if not arg:
                    continue
                yield _expandProjectArg(arg, filename_arg, for_eval=False)
            else:
                assert False, (command, line)

def _considerGithubWorkflowOptions(phase):
    if False:
        i = 10
        return i + 15
    try:
        github_option_index = sys.argv.index('--github-workflow-options')
    except ValueError:
        return
    import json
    early_names = ('main', 'script-name', 'enable-plugins', 'disable-plugins')

    def filterByName(key):
        if False:
            for i in range(10):
                print('nop')
        if key in ('nuitka-version', 'working-directory', 'access-token', 'disable-cache'):
            return False
        if key.startswith('macos-') and (not isMacOS()):
            return False
        if (key.startswith('windows-') or key == 'mingw64') and (not isWin32Windows()):
            return False
        if key.startswith('linux-') and (not isLinux()):
            return False
        if phase == 'early':
            return key in early_names
        else:
            return key not in early_names
    options_added = []
    for (key, value) in json.loads(os.environ['NUITKA_WORKFLOW_INPUTS']).items():
        if not value:
            continue
        if not filterByName(key):
            continue
        option_name = '--%s' % key
        if parser.isBooleanOption('--%s' % key):
            if value == 'false':
                continue
            options_added.append(option_name)
        else:
            if value == 'false':
                continue
            options_added.append('%s=%s' % (option_name, value))
    sys.argv = sys.argv[:github_option_index + (1 if phase == 'early' else 0)] + options_added + sys.argv[github_option_index + 1:]

def parseOptions(logger):
    if False:
        return 10
    extra_args = []
    if is_nuitka_run:
        count = 0
        for (count, arg) in enumerate(sys.argv):
            if count == 0:
                continue
            if arg[0] != '-':
                break
            if arg == '--':
                count += 1
                break
        if count > 0:
            extra_args = sys.argv[count + 1:]
            sys.argv = sys.argv[0:count + 1]
    filename_args = []
    module_mode = False
    for (count, arg) in enumerate(sys.argv):
        if count == 0:
            continue
        if arg.startswith('--main='):
            filename_args.append(arg)
        if arg == '--module':
            module_mode = True
        if arg[0] != '-':
            filename_args.append(arg)
            break
    for filename in filename_args:
        sys.argv = [sys.argv[0]] + list(getNuitkaProjectOptions(logger, filename, module_mode)) + sys.argv[1:]
    _considerGithubWorkflowOptions(phase='early')
    _considerPluginOptions(logger)
    _considerGithubWorkflowOptions(phase='late')
    (options, positional_args) = parser.parse_args()
    if not positional_args and (not options.mains) and (not parser.hasNonCompilingAction(options)):
        parser.print_help()
        logger.sysexit('\nError, need filename argument with python module or main program.')
    if not options.immediate_execution and len(positional_args) > 1:
        parser.print_help()
        logger.sysexit('\nError, specify only one positional argument unless "--run" is specified to\npass them to the compiled program execution.')
    return (is_nuitka_run, options, positional_args, extra_args)

def runSpecialCommandsFromOptions(options):
    if False:
        print('Hello World!')
    if options.plugin_list:
        from nuitka.plugins.Plugins import listPlugins
        listPlugins()
        sys.exit(0)
    if options.list_package_dlls:
        from nuitka.tools.scanning.DisplayPackageDLLs import displayDLLs
        displayDLLs(options.list_package_dlls)
        sys.exit(0)
    if options.list_package_data:
        from nuitka.tools.scanning.DisplayPackageData import displayPackageData
        displayPackageData(options.list_package_data)
        sys.exit(0)
    if options.edit_module_code:
        from nuitka.tools.general.find_module.FindModuleCode import editModuleCode
        editModuleCode(options.edit_module_code)
        sys.exit(0)
    if options.create_environment_from_report:
        from nuitka.tools.environments.CreateEnvironment import createEnvironmentFromReport
        createEnvironmentFromReport(environment_folder=os.path.expanduser(options.create_environment_from_report), report_filename=os.path.expanduser(options.compilation_report_filename))
        sys.exit(0)