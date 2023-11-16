"""Extension management for Windows.

Under Windows it is unlikely the .obj files are of use, as special compiler options
are needed (primarily to toggle the behavior of "public" symbols.

I don't consider it worth parsing the MSVC makefiles for compiler options.  Even if
we get it just right, a specific freeze application may have specific compiler
options anyway (eg, to enable or disable specific functionality)

So my basic strategy is:

* Have some Windows INI files which "describe" one or more extension modules.
  (Freeze comes with a default one for all known modules - but you can specify
  your own).
* This description can include:
  - The MSVC .dsp file for the extension.  The .c source file names
    are extracted from there.
  - Specific compiler/linker options
  - Flag to indicate if Unicode compilation is expected.

At the moment the name and location of this INI file is hardcoded,
but an obvious enhancement would be to provide command line options.
"""
import os, sys
try:
    import win32api
except ImportError:
    win32api = None

class CExtension:
    """An abstraction of an extension implemented in C/C++
    """

    def __init__(self, name, sourceFiles):
        if False:
            while True:
                i = 10
        self.name = name
        self.sourceFiles = sourceFiles
        self.compilerOptions = []
        self.linkerLibs = []

    def GetSourceFiles(self):
        if False:
            i = 10
            return i + 15
        return self.sourceFiles

    def AddCompilerOption(self, option):
        if False:
            print('Hello World!')
        self.compilerOptions.append(option)

    def GetCompilerOptions(self):
        if False:
            i = 10
            return i + 15
        return self.compilerOptions

    def AddLinkerLib(self, lib):
        if False:
            for i in range(10):
                print('nop')
        self.linkerLibs.append(lib)

    def GetLinkerLibs(self):
        if False:
            return 10
        return self.linkerLibs

def checkextensions(unknown, extra_inis, prefix):
    if False:
        print('Hello World!')
    defaultMapName = os.path.join(os.path.split(sys.argv[0])[0], 'extensions_win32.ini')
    if not os.path.isfile(defaultMapName):
        sys.stderr.write('WARNING: %s can not be found - standard extensions may not be found\n' % defaultMapName)
    else:
        extra_inis.append(defaultMapName)
    ret = []
    for mod in unknown:
        for ini in extra_inis:
            defn = get_extension_defn(mod, ini, prefix)
            if defn is not None:
                ret.append(defn)
                break
        else:
            sys.stderr.write('No definition of module %s in any specified map file.\n' % mod)
    return ret

def get_extension_defn(moduleName, mapFileName, prefix):
    if False:
        for i in range(10):
            print('nop')
    if win32api is None:
        return None
    os.environ['PYTHONPREFIX'] = prefix
    dsp = win32api.GetProfileVal(moduleName, 'dsp', '', mapFileName)
    if dsp == '':
        return None
    dsp = win32api.ExpandEnvironmentStrings(dsp)
    if not os.path.isabs(dsp):
        dsp = os.path.join(os.path.split(mapFileName)[0], dsp)
    sourceFiles = parse_dsp(dsp)
    if sourceFiles is None:
        return None
    module = CExtension(moduleName, sourceFiles)
    os.environ['dsp_path'] = os.path.split(dsp)[0]
    os.environ['ini_path'] = os.path.split(mapFileName)[0]
    cl_options = win32api.GetProfileVal(moduleName, 'cl', '', mapFileName)
    if cl_options:
        module.AddCompilerOption(win32api.ExpandEnvironmentStrings(cl_options))
    exclude = win32api.GetProfileVal(moduleName, 'exclude', '', mapFileName)
    exclude = exclude.split()
    if win32api.GetProfileVal(moduleName, 'Unicode', 0, mapFileName):
        module.AddCompilerOption('/D UNICODE /D _UNICODE')
    libs = win32api.GetProfileVal(moduleName, 'libs', '', mapFileName).split()
    for lib in libs:
        module.AddLinkerLib(win32api.ExpandEnvironmentStrings(lib))
    for exc in exclude:
        if exc in module.sourceFiles:
            module.sourceFiles.remove(exc)
    return module

def parse_dsp(dsp):
    if False:
        print('Hello World!')
    ret = []
    (dsp_path, dsp_name) = os.path.split(dsp)
    try:
        with open(dsp, 'r') as fp:
            lines = fp.readlines()
    except IOError as msg:
        sys.stderr.write('%s: %s\n' % (dsp, msg))
        return None
    for line in lines:
        fields = line.strip().split('=', 2)
        if fields[0] == 'SOURCE':
            if os.path.splitext(fields[1])[1].lower() in ['.cpp', '.c']:
                ret.append(win32api.GetFullPathName(os.path.join(dsp_path, fields[1])))
    return ret

def write_extension_table(fname, modules):
    if False:
        print('Hello World!')
    fp = open(fname, 'w')
    try:
        fp.write(ext_src_header)
        for module in modules:
            name = module.name.split('.')[-1]
            fp.write('extern void init%s(void);\n' % name)
        fp.write(ext_tab_header)
        for module in modules:
            name = module.name.split('.')[-1]
            fp.write('\t{"%s", init%s},\n' % (name, name))
        fp.write(ext_tab_footer)
        fp.write(ext_src_footer)
    finally:
        fp.close()
ext_src_header = '#include "Python.h"\n'
ext_tab_header = '\nstatic struct _inittab extensions[] = {\n'
ext_tab_footer = '        /* Sentinel */\n        {0, 0}\n};\n'
ext_src_footer = 'extern DL_IMPORT(int) PyImport_ExtendInittab(struct _inittab *newtab);\n\nint PyInitFrozenExtensions()\n{\n        return PyImport_ExtendInittab(extensions);\n}\n\n'