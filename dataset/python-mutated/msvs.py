"""SCons.Tool.msvs

Tool-specific initialization for Microsoft Visual Studio project files.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
from __future__ import print_function
__revision__ = 'src/engine/SCons/Tool/msvs.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.compat
import base64
import hashlib
import ntpath
import os
import pickle
import re
import sys
import SCons.Builder
import SCons.Node.FS
import SCons.Platform.win32
import SCons.Script.SConscript
import SCons.PathList
import SCons.Util
import SCons.Warnings
from .MSCommon import msvc_exists, msvc_setup_env_once
from SCons.Defaults import processDefines
from SCons.compat import PICKLE_PROTOCOL

def xmlify(s):
    if False:
        return 10
    s = s.replace('&', '&amp;')
    s = s.replace("'", '&apos;')
    s = s.replace('"', '&quot;')
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    s = s.replace('\n', '&#x0A;')
    return s

def processIncludes(includes, env, target, source):
    if False:
        return 10
    '\n    Process a CPPPATH list in includes, given the env, target and source.\n    Returns a list of directory paths. These paths are absolute so we avoid\n    putting pound-prefixed paths in a Visual Studio project file.\n    '
    return [env.Dir(i).abspath for i in SCons.PathList.PathList(includes).subst_path(env, target, source)]
external_makefile_guid = '{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}'

def _generateGUID(slnfile, name):
    if False:
        i = 10
        return i + 15
    'This generates a dummy GUID for the sln file to use.  It is\n    based on the MD5 signatures of the sln filename plus the name of\n    the project.  It basically just needs to be unique, and not\n    change with each invocation.'
    m = hashlib.md5()
    m.update(bytearray(ntpath.normpath(str(slnfile)) + str(name), 'utf-8'))
    solution = m.hexdigest().upper()
    solution = '{' + solution[:8] + '-' + solution[8:12] + '-' + solution[12:16] + '-' + solution[16:20] + '-' + solution[20:32] + '}'
    return solution
version_re = re.compile('(\\d+\\.\\d+)(.*)')

def msvs_parse_version(s):
    if False:
        i = 10
        return i + 15
    '\n    Split a Visual Studio version, which may in fact be something like\n    \'7.0Exp\', into is version number (returned as a float) and trailing\n    "suite" portion.\n    '
    (num, suite) = version_re.match(s).groups()
    return (float(num), suite)

def getExecScriptMain(env, xml=None):
    if False:
        while True:
            i = 10
    scons_home = env.get('SCONS_HOME')
    if not scons_home and 'SCONS_LIB_DIR' in os.environ:
        scons_home = os.environ['SCONS_LIB_DIR']
    if scons_home:
        exec_script_main = "from os.path import join; import sys; sys.path = [ r'%s' ] + sys.path; import SCons.Script; SCons.Script.main()" % scons_home
    else:
        version = SCons.__version__
        exec_script_main = "from os.path import join; import sys; sys.path = [ join(sys.prefix, 'Lib', 'site-packages', 'scons-%(version)s'), join(sys.prefix, 'scons-%(version)s'), join(sys.prefix, 'Lib', 'site-packages', 'scons'), join(sys.prefix, 'scons') ] + sys.path; import SCons.Script; SCons.Script.main()" % locals()
    if xml:
        exec_script_main = xmlify(exec_script_main)
    return exec_script_main
try:
    python_root = os.environ['PYTHON_ROOT']
except KeyError:
    python_executable = sys.executable
else:
    python_executable = os.path.join('$$(PYTHON_ROOT)', os.path.split(sys.executable)[1])

class Config(object):
    pass

def splitFully(path):
    if False:
        print('Hello World!')
    (dir, base) = os.path.split(path)
    if dir and dir != '' and (dir != path):
        return splitFully(dir) + [base]
    if base == '':
        return []
    return [base]

def makeHierarchy(sources):
    if False:
        for i in range(10):
            print('nop')
    'Break a list of files into a hierarchy; for each value, if it is a string,\n       then it is a file.  If it is a dictionary, it is a folder.  The string is\n       the original path of the file.'
    hierarchy = {}
    for file in sources:
        path = splitFully(file)
        if len(path):
            dict = hierarchy
            for part in path[:-1]:
                if part not in dict:
                    dict[part] = {}
                dict = dict[part]
            dict[path[-1]] = file
    return hierarchy

class _UserGenerator(object):
    """
    Base class for .dsp.user file generator
    """
    usrhead = None
    usrdebg = None
    usrconf = None
    createfile = False

    def __init__(self, dspfile, source, env):
        if False:
            for i in range(10):
                print('nop')
        if 'variant' not in env:
            raise SCons.Errors.InternalError("You must specify a 'variant' argument (i.e. 'Debug' or " + "'Release') to create an MSVSProject.")
        elif SCons.Util.is_String(env['variant']):
            variants = [env['variant']]
        elif SCons.Util.is_List(env['variant']):
            variants = env['variant']
        if 'DebugSettings' not in env or env['DebugSettings'] is None:
            dbg_settings = []
        elif SCons.Util.is_Dict(env['DebugSettings']):
            dbg_settings = [env['DebugSettings']]
        elif SCons.Util.is_List(env['DebugSettings']):
            if len(env['DebugSettings']) != len(variants):
                raise SCons.Errors.InternalError("Sizes of 'DebugSettings' and 'variant' lists must be the same.")
            dbg_settings = []
            for ds in env['DebugSettings']:
                if SCons.Util.is_Dict(ds):
                    dbg_settings.append(ds)
                else:
                    dbg_settings.append({})
        else:
            dbg_settings = []
        if len(dbg_settings) == 1:
            dbg_settings = dbg_settings * len(variants)
        self.createfile = self.usrhead and self.usrdebg and self.usrconf and dbg_settings and bool([ds for ds in dbg_settings if ds])
        if self.createfile:
            dbg_settings = dict(list(zip(variants, dbg_settings)))
            for (var, src) in dbg_settings.items():
                trg = {}
                for key in [k for k in list(self.usrdebg.keys()) if k in src]:
                    trg[key] = str(src[key])
                self.configs[var].debug = trg

    def UserHeader(self):
        if False:
            i = 10
            return i + 15
        encoding = self.env.subst('$MSVSENCODING')
        versionstr = self.versionstr
        self.usrfile.write(self.usrhead % locals())

    def UserProject(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def Build(self):
        if False:
            print('Hello World!')
        if not self.createfile:
            return
        try:
            filename = self.dspabs + '.user'
            self.usrfile = open(filename, 'w')
        except IOError as detail:
            raise SCons.Errors.InternalError('Unable to open "' + filename + '" for writing:' + str(detail))
        else:
            self.UserHeader()
            self.UserProject()
            self.usrfile.close()
V9UserHeader = '<?xml version="1.0" encoding="%(encoding)s"?>\n<VisualStudioUserFile\n\tProjectType="Visual C++"\n\tVersion="%(versionstr)s"\n\tShowAllFiles="false"\n\t>\n\t<Configurations>\n'
V9UserConfiguration = '\t\t<Configuration\n\t\t\tName="%(variant)s|%(platform)s"\n\t\t\t>\n\t\t\t<DebugSettings\n%(debug_settings)s\n\t\t\t/>\n\t\t</Configuration>\n'
V9DebugSettings = {'Command': '$(TargetPath)', 'WorkingDirectory': None, 'CommandArguments': None, 'Attach': 'false', 'DebuggerType': '3', 'Remote': '1', 'RemoteMachine': None, 'RemoteCommand': None, 'HttpUrl': None, 'PDBPath': None, 'SQLDebugging': None, 'Environment': None, 'EnvironmentMerge': 'true', 'DebuggerFlavor': None, 'MPIRunCommand': None, 'MPIRunArguments': None, 'MPIRunWorkingDirectory': None, 'ApplicationCommand': None, 'ApplicationArguments': None, 'ShimCommand': None, 'MPIAcceptMode': None, 'MPIAcceptFilter': None}

class _GenerateV7User(_UserGenerator):
    """Generates a Project file for MSVS .NET"""

    def __init__(self, dspfile, source, env):
        if False:
            while True:
                i = 10
        if self.version_num >= 9.0:
            self.usrhead = V9UserHeader
            self.usrconf = V9UserConfiguration
            self.usrdebg = V9DebugSettings
        _UserGenerator.__init__(self, dspfile, source, env)

    def UserProject(self):
        if False:
            for i in range(10):
                print('nop')
        confkeys = sorted(self.configs.keys())
        for kind in confkeys:
            variant = self.configs[kind].variant
            platform = self.configs[kind].platform
            debug = self.configs[kind].debug
            if debug:
                debug_settings = '\n'.join(['\t\t\t\t%s="%s"' % (key, xmlify(value)) for (key, value) in debug.items() if value is not None])
                self.usrfile.write(self.usrconf % locals())
        self.usrfile.write('\t</Configurations>\n</VisualStudioUserFile>')
V10UserHeader = '<?xml version="1.0" encoding="%(encoding)s"?>\n<Project ToolsVersion="%(versionstr)s" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n'
V10UserConfiguration = '\t<PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">\n%(debug_settings)s\n\t</PropertyGroup>\n'
V10DebugSettings = {'LocalDebuggerCommand': None, 'LocalDebuggerCommandArguments': None, 'LocalDebuggerEnvironment': None, 'DebuggerFlavor': 'WindowsLocalDebugger', 'LocalDebuggerWorkingDirectory': None, 'LocalDebuggerAttach': None, 'LocalDebuggerDebuggerType': None, 'LocalDebuggerMergeEnvironment': None, 'LocalDebuggerSQLDebugging': None, 'RemoteDebuggerCommand': None, 'RemoteDebuggerCommandArguments': None, 'RemoteDebuggerWorkingDirectory': None, 'RemoteDebuggerServerName': None, 'RemoteDebuggerConnection': None, 'RemoteDebuggerDebuggerType': None, 'RemoteDebuggerAttach': None, 'RemoteDebuggerSQLDebugging': None, 'DeploymentDirectory': None, 'AdditionalFiles': None, 'RemoteDebuggerDeployDebugCppRuntime': None, 'WebBrowserDebuggerHttpUrl': None, 'WebBrowserDebuggerDebuggerType': None, 'WebServiceDebuggerHttpUrl': None, 'WebServiceDebuggerDebuggerType': None, 'WebServiceDebuggerSQLDebugging': None}

class _GenerateV10User(_UserGenerator):
    """Generates a Project'user file for MSVS 2010 or later"""

    def __init__(self, dspfile, source, env):
        if False:
            i = 10
            return i + 15
        (version_num, suite) = msvs_parse_version(env['MSVS_VERSION'])
        if version_num >= 14.2:
            self.versionstr = '16.0'
        elif version_num >= 14.1:
            self.versionstr = '15.0'
        elif version_num == 14.0:
            self.versionstr = '14.0'
        else:
            self.versionstr = '4.0'
        self.usrhead = V10UserHeader
        self.usrconf = V10UserConfiguration
        self.usrdebg = V10DebugSettings
        _UserGenerator.__init__(self, dspfile, source, env)

    def UserProject(self):
        if False:
            return 10
        confkeys = sorted(self.configs.keys())
        for kind in confkeys:
            variant = self.configs[kind].variant
            platform = self.configs[kind].platform
            debug = self.configs[kind].debug
            if debug:
                debug_settings = '\n'.join(['\t\t<%s>%s</%s>' % (key, xmlify(value), key) for (key, value) in debug.items() if value is not None])
                self.usrfile.write(self.usrconf % locals())
        self.usrfile.write('</Project>')

class _DSPGenerator(object):
    """ Base class for DSP generators """
    srcargs = ['srcs', 'incs', 'localincs', 'resources', 'misc']

    def __init__(self, dspfile, source, env):
        if False:
            i = 10
            return i + 15
        self.dspfile = str(dspfile)
        try:
            get_abspath = dspfile.get_abspath
        except AttributeError:
            self.dspabs = os.path.abspath(dspfile)
        else:
            self.dspabs = get_abspath()
        if 'variant' not in env:
            raise SCons.Errors.InternalError("You must specify a 'variant' argument (i.e. 'Debug' or " + "'Release') to create an MSVSProject.")
        elif SCons.Util.is_String(env['variant']):
            variants = [env['variant']]
        elif SCons.Util.is_List(env['variant']):
            variants = env['variant']
        if 'buildtarget' not in env or env['buildtarget'] is None:
            buildtarget = ['']
        elif SCons.Util.is_String(env['buildtarget']):
            buildtarget = [env['buildtarget']]
        elif SCons.Util.is_List(env['buildtarget']):
            if len(env['buildtarget']) != len(variants):
                raise SCons.Errors.InternalError("Sizes of 'buildtarget' and 'variant' lists must be the same.")
            buildtarget = []
            for bt in env['buildtarget']:
                if SCons.Util.is_String(bt):
                    buildtarget.append(bt)
                else:
                    buildtarget.append(bt.get_abspath())
        else:
            buildtarget = [env['buildtarget'].get_abspath()]
        if len(buildtarget) == 1:
            bt = buildtarget[0]
            buildtarget = []
            for _ in variants:
                buildtarget.append(bt)
        if 'outdir' not in env or env['outdir'] is None:
            outdir = ['']
        elif SCons.Util.is_String(env['outdir']):
            outdir = [env['outdir']]
        elif SCons.Util.is_List(env['outdir']):
            if len(env['outdir']) != len(variants):
                raise SCons.Errors.InternalError("Sizes of 'outdir' and 'variant' lists must be the same.")
            outdir = []
            for s in env['outdir']:
                if SCons.Util.is_String(s):
                    outdir.append(s)
                else:
                    outdir.append(s.get_abspath())
        else:
            outdir = [env['outdir'].get_abspath()]
        if len(outdir) == 1:
            s = outdir[0]
            outdir = []
            for v in variants:
                outdir.append(s)
        if 'runfile' not in env or env['runfile'] is None:
            runfile = buildtarget[-1:]
        elif SCons.Util.is_String(env['runfile']):
            runfile = [env['runfile']]
        elif SCons.Util.is_List(env['runfile']):
            if len(env['runfile']) != len(variants):
                raise SCons.Errors.InternalError("Sizes of 'runfile' and 'variant' lists must be the same.")
            runfile = []
            for s in env['runfile']:
                if SCons.Util.is_String(s):
                    runfile.append(s)
                else:
                    runfile.append(s.get_abspath())
        else:
            runfile = [env['runfile'].get_abspath()]
        if len(runfile) == 1:
            s = runfile[0]
            runfile = []
            for v in variants:
                runfile.append(s)
        self.sconscript = env['MSVSSCONSCRIPT']

        def GetKeyFromEnv(env, key, variants):
            if False:
                print('Hello World!')
            '\n            Retrieves a specific key from the environment. If the key is\n            present, it is expected to either be a string or a list with length\n            equal to the number of variants. The function returns a list of\n            the desired value (e.g. cpp include paths) guaranteed to be of\n            length equal to the length of the variants list.\n            '
            if key not in env or env[key] is None:
                return [''] * len(variants)
            elif SCons.Util.is_String(env[key]):
                return [env[key]] * len(variants)
            elif SCons.Util.is_List(env[key]):
                if len(env[key]) != len(variants):
                    raise SCons.Errors.InternalError("Sizes of '%s' and 'variant' lists must be the same." % key)
                else:
                    return env[key]
            else:
                raise SCons.Errors.InternalError("Unsupported type for key '%s' in environment: %s" % (key, type(env[key])))
        cmdargs = GetKeyFromEnv(env, 'cmdargs', variants)
        if 'cppdefines' in env:
            cppdefines = GetKeyFromEnv(env, 'cppdefines', variants)
        else:
            cppdefines = [env.get('CPPDEFINES', [])] * len(variants)
        if 'cpppaths' in env:
            cpppaths = GetKeyFromEnv(env, 'cpppaths', variants)
        else:
            cpppaths = [env.get('CPPPATH', [])] * len(variants)
        self.env = env
        if 'name' in self.env:
            self.name = self.env['name']
        else:
            self.name = os.path.basename(SCons.Util.splitext(self.dspfile)[0])
        self.name = self.env.subst(self.name)
        sourcenames = ['Source Files', 'Header Files', 'Local Headers', 'Resource Files', 'Other Files']
        self.sources = {}
        for n in sourcenames:
            self.sources[n] = []
        self.configs = {}
        self.nokeep = 0
        if 'nokeep' in env and env['variant'] != 0:
            self.nokeep = 1
        if self.nokeep == 0 and os.path.exists(self.dspabs):
            self.Parse()
        for t in zip(sourcenames, self.srcargs):
            if t[1] in self.env:
                if SCons.Util.is_List(self.env[t[1]]):
                    for i in self.env[t[1]]:
                        if not i in self.sources[t[0]]:
                            self.sources[t[0]].append(i)
                elif not self.env[t[1]] in self.sources[t[0]]:
                    self.sources[t[0]].append(self.env[t[1]])
        for n in sourcenames:
            self.sources[n].sort(key=lambda a: a.lower())

        def AddConfig(self, variant, buildtarget, outdir, runfile, cmdargs, cppdefines, cpppaths, dspfile=dspfile, env=env):
            if False:
                for i in range(10):
                    print('nop')
            config = Config()
            config.buildtarget = buildtarget
            config.outdir = outdir
            config.cmdargs = cmdargs
            config.cppdefines = cppdefines
            config.runfile = runfile
            config.cpppaths = processIncludes(cpppaths, env, None, None)
            match = re.match('(.*)\\|(.*)', variant)
            if match:
                config.variant = match.group(1)
                config.platform = match.group(2)
            else:
                config.variant = variant
                config.platform = 'Win32'
            self.configs[variant] = config
            print("Adding '" + self.name + ' - ' + config.variant + '|' + config.platform + "' to '" + str(dspfile) + "'")
        for i in range(len(variants)):
            AddConfig(self, variants[i], buildtarget[i], outdir[i], runfile[i], cmdargs[i], cppdefines[i], cpppaths[i])
        self.platforms = []
        for key in list(self.configs.keys()):
            platform = self.configs[key].platform
            if platform not in self.platforms:
                self.platforms.append(platform)

    def Build(self):
        if False:
            print('Hello World!')
        pass
V6DSPHeader = '# Microsoft Developer Studio Project File - Name="%(name)s" - Package Owner=<4>\n# Microsoft Developer Studio Generated Build File, Format Version 6.00\n# ** DO NOT EDIT **\n\n# TARGTYPE "Win32 (x86) External Target" 0x0106\n\nCFG=%(name)s - Win32 %(confkey)s\n!MESSAGE This is not a valid makefile. To build this project using NMAKE,\n!MESSAGE use the Export Makefile command and run\n!MESSAGE\n!MESSAGE NMAKE /f "%(name)s.mak".\n!MESSAGE\n!MESSAGE You can specify a configuration when running NMAKE\n!MESSAGE by defining the macro CFG on the command line. For example:\n!MESSAGE\n!MESSAGE NMAKE /f "%(name)s.mak" CFG="%(name)s - Win32 %(confkey)s"\n!MESSAGE\n!MESSAGE Possible choices for configuration are:\n!MESSAGE\n'

class _GenerateV6DSP(_DSPGenerator):
    """Generates a Project file for MSVS 6.0"""

    def PrintHeader(self):
        if False:
            for i in range(10):
                print('nop')
        confkeys = sorted(self.configs.keys())
        name = self.name
        confkey = confkeys[0]
        self.file.write(V6DSPHeader % locals())
        for kind in confkeys:
            self.file.write('!MESSAGE "%s - Win32 %s" (based on "Win32 (x86) External Target")\n' % (name, kind))
        self.file.write('!MESSAGE\n\n')

    def PrintProject(self):
        if False:
            return 10
        name = self.name
        self.file.write('# Begin Project\n# PROP AllowPerConfigDependencies 0\n# PROP Scc_ProjName ""\n# PROP Scc_LocalPath ""\n\n')
        first = 1
        confkeys = sorted(self.configs.keys())
        for kind in confkeys:
            outdir = self.configs[kind].outdir
            buildtarget = self.configs[kind].buildtarget
            if first == 1:
                self.file.write('!IF  "$(CFG)" == "%s - Win32 %s"\n\n' % (name, kind))
                first = 0
            else:
                self.file.write('\n!ELSEIF  "$(CFG)" == "%s - Win32 %s"\n\n' % (name, kind))
            env_has_buildtarget = 'MSVSBUILDTARGET' in self.env
            if not env_has_buildtarget:
                self.env['MSVSBUILDTARGET'] = buildtarget
            for base in ('BASE ', ''):
                self.file.write('# PROP %sUse_MFC 0\n# PROP %sUse_Debug_Libraries ' % (base, base))
                if 'debug' not in kind.lower():
                    self.file.write('0\n')
                else:
                    self.file.write('1\n')
                self.file.write('# PROP %sOutput_Dir "%s"\n# PROP %sIntermediate_Dir "%s"\n' % (base, outdir, base, outdir))
                cmd = 'echo Starting SCons && ' + self.env.subst('$MSVSBUILDCOM', 1)
                self.file.write('# PROP %sCmd_Line "%s"\n# PROP %sRebuild_Opt "-c && %s"\n# PROP %sTarget_File "%s"\n# PROP %sBsc_Name ""\n# PROP %sTarget_Dir ""\n' % (base, cmd, base, cmd, base, buildtarget, base, base))
            if not env_has_buildtarget:
                del self.env['MSVSBUILDTARGET']
        self.file.write('\n!ENDIF\n\n# Begin Target\n\n')
        for kind in confkeys:
            self.file.write('# Name "%s - Win32 %s"\n' % (name, kind))
        self.file.write('\n')
        first = 0
        for kind in confkeys:
            if first == 0:
                self.file.write('!IF  "$(CFG)" == "%s - Win32 %s"\n\n' % (name, kind))
                first = 1
            else:
                self.file.write('!ELSEIF  "$(CFG)" == "%s - Win32 %s"\n\n' % (name, kind))
        self.file.write('!ENDIF\n\n')
        self.PrintSourceFiles()
        self.file.write('# End Target\n# End Project\n')
        if self.nokeep == 0:
            pdata = pickle.dumps(self.configs, PICKLE_PROTOCOL)
            pdata = base64.b64encode(pdata).decode()
            self.file.write(pdata + '\n')
            pdata = pickle.dumps(self.sources, PICKLE_PROTOCOL)
            pdata = base64.b64encode(pdata).decode()
            self.file.write(pdata + '\n')

    def PrintSourceFiles(self):
        if False:
            while True:
                i = 10
        categories = {'Source Files': 'cpp|c|cxx|l|y|def|odl|idl|hpj|bat', 'Header Files': 'h|hpp|hxx|hm|inl', 'Local Headers': 'h|hpp|hxx|hm|inl', 'Resource Files': 'r|rc|ico|cur|bmp|dlg|rc2|rct|bin|cnt|rtf|gif|jpg|jpeg|jpe', 'Other Files': ''}
        for kind in sorted(list(categories.keys()), key=lambda a: a.lower()):
            if not self.sources[kind]:
                continue
            self.file.write('# Begin Group "' + kind + '"\n\n')
            typelist = categories[kind].replace('|', ';')
            self.file.write('# PROP Default_Filter "' + typelist + '"\n')
            for file in self.sources[kind]:
                file = os.path.normpath(file)
                self.file.write('# Begin Source File\n\nSOURCE="' + file + '"\n# End Source File\n')
            self.file.write('# End Group\n')
        self.file.write('# Begin Source File\n\nSOURCE="' + str(self.sconscript) + '"\n# End Source File\n')

    def Parse(self):
        if False:
            i = 10
            return i + 15
        try:
            dspfile = open(self.dspabs, 'r')
        except IOError:
            return
        line = dspfile.readline()
        while line:
            if '# End Project' in line:
                break
            line = dspfile.readline()
        line = dspfile.readline()
        datas = line
        while line and line != '\n':
            line = dspfile.readline()
            datas = datas + line
        try:
            datas = base64.decodestring(datas)
            data = pickle.loads(datas)
        except KeyboardInterrupt:
            raise
        except:
            return
        self.configs.update(data)
        data = None
        line = dspfile.readline()
        datas = line
        while line and line != '\n':
            line = dspfile.readline()
            datas = datas + line
        dspfile.close()
        try:
            datas = base64.decodestring(datas)
            data = pickle.loads(datas)
        except KeyboardInterrupt:
            raise
        except:
            return
        self.sources.update(data)

    def Build(self):
        if False:
            while True:
                i = 10
        try:
            self.file = open(self.dspabs, 'w')
        except IOError as detail:
            raise SCons.Errors.InternalError('Unable to open "' + self.dspabs + '" for writing:' + str(detail))
        else:
            self.PrintHeader()
            self.PrintProject()
            self.file.close()
V7DSPHeader = '<?xml version="1.0" encoding="%(encoding)s"?>\n<VisualStudioProject\n\tProjectType="Visual C++"\n\tVersion="%(versionstr)s"\n\tName="%(name)s"\n\tProjectGUID="%(project_guid)s"\n%(scc_attrs)s\n\tKeyword="MakeFileProj">\n'
V7DSPConfiguration = '\t\t<Configuration\n\t\t\tName="%(variant)s|%(platform)s"\n\t\t\tOutputDirectory="%(outdir)s"\n\t\t\tIntermediateDirectory="%(outdir)s"\n\t\t\tConfigurationType="0"\n\t\t\tUseOfMFC="0"\n\t\t\tATLMinimizesCRunTimeLibraryUsage="FALSE">\n\t\t\t<Tool\n\t\t\t\tName="VCNMakeTool"\n\t\t\t\tBuildCommandLine="%(buildcmd)s"\n\t\t\t\tReBuildCommandLine="%(rebuildcmd)s"\n\t\t\t\tCleanCommandLine="%(cleancmd)s"\n\t\t\t\tOutput="%(runfile)s"/>\n\t\t</Configuration>\n'
V8DSPHeader = '<?xml version="1.0" encoding="%(encoding)s"?>\n<VisualStudioProject\n\tProjectType="Visual C++"\n\tVersion="%(versionstr)s"\n\tName="%(name)s"\n\tProjectGUID="%(project_guid)s"\n\tRootNamespace="%(name)s"\n%(scc_attrs)s\n\tKeyword="MakeFileProj">\n'
V8DSPConfiguration = '\t\t<Configuration\n\t\t\tName="%(variant)s|%(platform)s"\n\t\t\tConfigurationType="0"\n\t\t\tUseOfMFC="0"\n\t\t\tATLMinimizesCRunTimeLibraryUsage="false"\n\t\t\t>\n\t\t\t<Tool\n\t\t\t\tName="VCNMakeTool"\n\t\t\t\tBuildCommandLine="%(buildcmd)s"\n\t\t\t\tReBuildCommandLine="%(rebuildcmd)s"\n\t\t\t\tCleanCommandLine="%(cleancmd)s"\n\t\t\t\tOutput="%(runfile)s"\n\t\t\t\tPreprocessorDefinitions="%(preprocdefs)s"\n\t\t\t\tIncludeSearchPath="%(includepath)s"\n\t\t\t\tForcedIncludes=""\n\t\t\t\tAssemblySearchPath=""\n\t\t\t\tForcedUsingAssemblies=""\n\t\t\t\tCompileAsManaged=""\n\t\t\t/>\n\t\t</Configuration>\n'

class _GenerateV7DSP(_DSPGenerator, _GenerateV7User):
    """Generates a Project file for MSVS .NET"""

    def __init__(self, dspfile, source, env):
        if False:
            i = 10
            return i + 15
        _DSPGenerator.__init__(self, dspfile, source, env)
        self.version = env['MSVS_VERSION']
        (self.version_num, self.suite) = msvs_parse_version(self.version)
        if self.version_num >= 9.0:
            self.versionstr = '9.00'
            self.dspheader = V8DSPHeader
            self.dspconfiguration = V8DSPConfiguration
        elif self.version_num >= 8.0:
            self.versionstr = '8.00'
            self.dspheader = V8DSPHeader
            self.dspconfiguration = V8DSPConfiguration
        else:
            if self.version_num >= 7.1:
                self.versionstr = '7.10'
            else:
                self.versionstr = '7.00'
            self.dspheader = V7DSPHeader
            self.dspconfiguration = V7DSPConfiguration
        self.file = None
        _GenerateV7User.__init__(self, dspfile, source, env)

    def PrintHeader(self):
        if False:
            print('Hello World!')
        env = self.env
        versionstr = self.versionstr
        name = self.name
        encoding = self.env.subst('$MSVSENCODING')
        scc_provider = env.get('MSVS_SCC_PROVIDER', '')
        scc_project_name = env.get('MSVS_SCC_PROJECT_NAME', '')
        scc_aux_path = env.get('MSVS_SCC_AUX_PATH', '')
        scc_local_path_legacy = env.get('MSVS_SCC_LOCAL_PATH', '')
        scc_connection_root = env.get('MSVS_SCC_CONNECTION_ROOT', os.curdir)
        scc_local_path = os.path.relpath(scc_connection_root, os.path.dirname(self.dspabs))
        project_guid = env.get('MSVS_PROJECT_GUID', '')
        if not project_guid:
            project_guid = _generateGUID(self.dspfile, '')
        if scc_provider != '':
            scc_attrs = '\tSccProjectName="%s"\n' % scc_project_name
            if scc_aux_path != '':
                scc_attrs += '\tSccAuxPath="%s"\n' % scc_aux_path
            scc_attrs += '\tSccLocalPath="%s"\n\tSccProvider="%s"' % (scc_local_path, scc_provider)
        elif scc_local_path_legacy != '':
            scc_attrs = '\tSccProjectName="%s"\n\tSccLocalPath="%s"' % (scc_project_name, scc_local_path_legacy)
        else:
            self.dspheader = self.dspheader.replace('%(scc_attrs)s\n', '')
        self.file.write(self.dspheader % locals())
        self.file.write('\t<Platforms>\n')
        for platform in self.platforms:
            self.file.write('\t\t<Platform\n\t\t\tName="%s"/>\n' % platform)
        self.file.write('\t</Platforms>\n')
        if self.version_num >= 8.0:
            self.file.write('\t<ToolFiles>\n\t</ToolFiles>\n')

    def PrintProject(self):
        if False:
            for i in range(10):
                print('nop')
        self.file.write('\t<Configurations>\n')
        confkeys = sorted(self.configs.keys())
        for kind in confkeys:
            variant = self.configs[kind].variant
            platform = self.configs[kind].platform
            outdir = self.configs[kind].outdir
            buildtarget = self.configs[kind].buildtarget
            runfile = self.configs[kind].runfile
            cmdargs = self.configs[kind].cmdargs
            cpppaths = self.configs[kind].cpppaths
            cppdefines = self.configs[kind].cppdefines
            env_has_buildtarget = 'MSVSBUILDTARGET' in self.env
            if not env_has_buildtarget:
                self.env['MSVSBUILDTARGET'] = buildtarget
            starting = 'echo Starting SCons && '
            if cmdargs:
                cmdargs = ' ' + cmdargs
            else:
                cmdargs = ''
            buildcmd = xmlify(starting + self.env.subst('$MSVSBUILDCOM', 1) + cmdargs)
            rebuildcmd = xmlify(starting + self.env.subst('$MSVSREBUILDCOM', 1) + cmdargs)
            cleancmd = xmlify(starting + self.env.subst('$MSVSCLEANCOM', 1) + cmdargs)
            preprocdefs = xmlify(';'.join(processDefines(cppdefines)))
            includepath = xmlify(';'.join(processIncludes(cpppaths, self.env, None, None)))
            if not env_has_buildtarget:
                del self.env['MSVSBUILDTARGET']
            self.file.write(self.dspconfiguration % locals())
        self.file.write('\t</Configurations>\n')
        if self.version_num >= 7.1:
            self.file.write('\t<References>\n\t</References>\n')
        self.PrintSourceFiles()
        self.file.write('</VisualStudioProject>\n')
        if self.nokeep == 0:
            pdata = pickle.dumps(self.configs, PICKLE_PROTOCOL)
            pdata = base64.b64encode(pdata).decode()
            self.file.write('<!-- SCons Data:\n' + pdata + '\n')
            pdata = pickle.dumps(self.sources, PICKLE_PROTOCOL)
            pdata = base64.b64encode(pdata).decode()
            self.file.write(pdata + '-->\n')

    def printSources(self, hierarchy, commonprefix):
        if False:
            while True:
                i = 10
        sorteditems = sorted(hierarchy.items(), key=lambda a: a[0].lower())
        for (key, value) in sorteditems:
            if SCons.Util.is_Dict(value):
                self.file.write('\t\t\t<Filter\n\t\t\t\tName="%s"\n\t\t\t\tFilter="">\n' % key)
                self.printSources(value, commonprefix)
                self.file.write('\t\t\t</Filter>\n')
        for (key, value) in sorteditems:
            if SCons.Util.is_String(value):
                file = value
                if commonprefix:
                    file = os.path.join(commonprefix, value)
                file = os.path.normpath(file)
                self.file.write('\t\t\t<File\n\t\t\t\tRelativePath="%s">\n\t\t\t</File>\n' % file)

    def PrintSourceFiles(self):
        if False:
            return 10
        categories = {'Source Files': 'cpp;c;cxx;l;y;def;odl;idl;hpj;bat', 'Header Files': 'h;hpp;hxx;hm;inl', 'Local Headers': 'h;hpp;hxx;hm;inl', 'Resource Files': 'r;rc;ico;cur;bmp;dlg;rc2;rct;bin;cnt;rtf;gif;jpg;jpeg;jpe', 'Other Files': ''}
        self.file.write('\t<Files>\n')
        cats = sorted([k for k in list(categories.keys()) if self.sources[k]], key=lambda a: a.lower())
        for kind in cats:
            if len(cats) > 1:
                self.file.write('\t\t<Filter\n\t\t\tName="%s"\n\t\t\tFilter="%s">\n' % (kind, categories[kind]))
            sources = self.sources[kind]
            commonprefix = None
            s = list(map(os.path.normpath, sources))
            cp = os.path.dirname(os.path.commonprefix(s))
            if cp and s[0][len(cp)] == os.sep:
                sources = [s[len(cp) + 1:] for s in sources]
                commonprefix = cp
            hierarchy = makeHierarchy(sources)
            self.printSources(hierarchy, commonprefix=commonprefix)
            if len(cats) > 1:
                self.file.write('\t\t</Filter>\n')
        self.file.write('\t\t<File\n\t\t\tRelativePath="%s">\n\t\t</File>\n' % str(self.sconscript))
        self.file.write('\t</Files>\n\t<Globals>\n\t</Globals>\n')

    def Parse(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            dspfile = open(self.dspabs, 'r')
        except IOError:
            return
        line = dspfile.readline()
        while line:
            if '<!-- SCons Data:' in line:
                break
            line = dspfile.readline()
        line = dspfile.readline()
        datas = line
        while line and line != '\n':
            line = dspfile.readline()
            datas = datas + line
        try:
            datas = base64.decodestring(datas)
            data = pickle.loads(datas)
        except KeyboardInterrupt:
            raise
        except:
            return
        self.configs.update(data)
        data = None
        line = dspfile.readline()
        datas = line
        while line and line != '\n':
            line = dspfile.readline()
            datas = datas + line
        dspfile.close()
        try:
            datas = base64.decodestring(datas)
            data = pickle.loads(datas)
        except KeyboardInterrupt:
            raise
        except:
            return
        self.sources.update(data)

    def Build(self):
        if False:
            print('Hello World!')
        try:
            self.file = open(self.dspabs, 'w')
        except IOError as detail:
            raise SCons.Errors.InternalError('Unable to open "' + self.dspabs + '" for writing:' + str(detail))
        else:
            self.PrintHeader()
            self.PrintProject()
            self.file.close()
        _GenerateV7User.Build(self)
V10DSPHeader = '<?xml version="1.0" encoding="%(encoding)s"?>\n<Project DefaultTargets="Build" ToolsVersion="%(versionstr)s" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n'
V10DSPProjectConfiguration = '\t\t<ProjectConfiguration Include="%(variant)s|%(platform)s">\n\t\t\t<Configuration>%(variant)s</Configuration>\n\t\t\t<Platform>%(platform)s</Platform>\n\t\t</ProjectConfiguration>\n'
V10DSPGlobals = '\t<PropertyGroup Label="Globals">\n\t\t<ProjectGuid>%(project_guid)s</ProjectGuid>\n%(scc_attrs)s\t\t<RootNamespace>%(name)s</RootNamespace>\n\t\t<Keyword>MakeFileProj</Keyword>\n\t\t<VCProjectUpgraderObjectName>NoUpgrade</VCProjectUpgraderObjectName>\n\t</PropertyGroup>\n'
V10DSPPropertyGroupCondition = '\t<PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'" Label="Configuration">\n\t\t<ConfigurationType>Makefile</ConfigurationType>\n\t\t<UseOfMfc>false</UseOfMfc>\n\t\t<PlatformToolset>%(toolset)s</PlatformToolset>\n\t</PropertyGroup>\n'
V10DSPImportGroupCondition = '\t<ImportGroup Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'" Label="PropertySheets">\n\t\t<Import Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" Condition="exists(\'$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props\')" Label="LocalAppDataPlatform" />\n\t</ImportGroup>\n'
V10DSPCommandLine = '\t\t<NMakeBuildCommandLine Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">%(buildcmd)s</NMakeBuildCommandLine>\n\t\t<NMakeReBuildCommandLine Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">%(rebuildcmd)s</NMakeReBuildCommandLine>\n\t\t<NMakeCleanCommandLine Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">%(cleancmd)s</NMakeCleanCommandLine>\n\t\t<NMakeOutput Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">%(runfile)s</NMakeOutput>\n\t\t<NMakePreprocessorDefinitions Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">%(preprocdefs)s</NMakePreprocessorDefinitions>\n\t\t<NMakeIncludeSearchPath Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">%(includepath)s</NMakeIncludeSearchPath>\n\t\t<NMakeForcedIncludes Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">$(NMakeForcedIncludes)</NMakeForcedIncludes>\n\t\t<NMakeAssemblySearchPath Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">$(NMakeAssemblySearchPath)</NMakeAssemblySearchPath>\n\t\t<NMakeForcedUsingAssemblies Condition="\'$(Configuration)|$(Platform)\'==\'%(variant)s|%(platform)s\'">$(NMakeForcedUsingAssemblies)</NMakeForcedUsingAssemblies>\n'
V15DSPHeader = '<?xml version="1.0" encoding="%(encoding)s"?>\n<Project DefaultTargets="Build" ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n'

class _GenerateV10DSP(_DSPGenerator, _GenerateV10User):
    """Generates a Project file for MSVS 2010"""

    def __init__(self, dspfile, source, env):
        if False:
            return 10
        _DSPGenerator.__init__(self, dspfile, source, env)
        self.dspheader = V10DSPHeader
        self.dspconfiguration = V10DSPProjectConfiguration
        self.dspglobals = V10DSPGlobals
        _GenerateV10User.__init__(self, dspfile, source, env)

    def PrintHeader(self):
        if False:
            i = 10
            return i + 15
        env = self.env
        name = self.name
        versionstr = self.versionstr
        encoding = env.subst('$MSVSENCODING')
        project_guid = env.get('MSVS_PROJECT_GUID', '')
        scc_provider = env.get('MSVS_SCC_PROVIDER', '')
        scc_project_name = env.get('MSVS_SCC_PROJECT_NAME', '')
        scc_aux_path = env.get('MSVS_SCC_AUX_PATH', '')
        scc_local_path_legacy = env.get('MSVS_SCC_LOCAL_PATH', '')
        scc_connection_root = env.get('MSVS_SCC_CONNECTION_ROOT', os.curdir)
        scc_local_path = os.path.relpath(scc_connection_root, os.path.dirname(self.dspabs))
        if not project_guid:
            project_guid = _generateGUID(self.dspfile, '')
        if scc_provider != '':
            scc_attrs = '\t\t<SccProjectName>%s</SccProjectName>\n' % scc_project_name
            if scc_aux_path != '':
                scc_attrs += '\t\t<SccAuxPath>%s</SccAuxPath>\n' % scc_aux_path
            scc_attrs += '\t\t<SccLocalPath>%s</SccLocalPath>\n\t\t<SccProvider>%s</SccProvider>\n' % (scc_local_path, scc_provider)
        elif scc_local_path_legacy != '':
            scc_attrs = '\t\t<SccProjectName>%s</SccProjectName>\n\t\t<SccLocalPath>%s</SccLocalPath>\n' % (scc_project_name, scc_local_path_legacy)
        else:
            self.dspglobals = self.dspglobals.replace('%(scc_attrs)s', '')
        self.file.write(self.dspheader % locals())
        self.file.write('\t<ItemGroup Label="ProjectConfigurations">\n')
        confkeys = sorted(self.configs.keys())
        for kind in confkeys:
            variant = self.configs[kind].variant
            platform = self.configs[kind].platform
            self.file.write(self.dspconfiguration % locals())
        self.file.write('\t</ItemGroup>\n')
        self.file.write(self.dspglobals % locals())

    def PrintProject(self):
        if False:
            while True:
                i = 10
        name = self.name
        confkeys = sorted(self.configs.keys())
        self.file.write('\t<Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />\n')
        toolset = ''
        if 'MSVC_VERSION' in self.env:
            (version_num, suite) = msvs_parse_version(self.env['MSVC_VERSION'])
            toolset = 'v%d' % (version_num * 10)
        for kind in confkeys:
            variant = self.configs[kind].variant
            platform = self.configs[kind].platform
            self.file.write(V10DSPPropertyGroupCondition % locals())
        self.file.write('\t<Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />\n')
        self.file.write('\t<ImportGroup Label="ExtensionSettings">\n')
        self.file.write('\t</ImportGroup>\n')
        for kind in confkeys:
            variant = self.configs[kind].variant
            platform = self.configs[kind].platform
            self.file.write(V10DSPImportGroupCondition % locals())
        self.file.write('\t<PropertyGroup Label="UserMacros" />\n')
        self.file.write('\t<PropertyGroup>\n')
        self.file.write('\t<_ProjectFileVersion>10.0.30319.1</_ProjectFileVersion>\n')
        for kind in confkeys:
            variant = self.configs[kind].variant
            platform = self.configs[kind].platform
            outdir = self.configs[kind].outdir
            buildtarget = self.configs[kind].buildtarget
            runfile = self.configs[kind].runfile
            cmdargs = self.configs[kind].cmdargs
            cpppaths = self.configs[kind].cpppaths
            cppdefines = self.configs[kind].cppdefines
            env_has_buildtarget = 'MSVSBUILDTARGET' in self.env
            if not env_has_buildtarget:
                self.env['MSVSBUILDTARGET'] = buildtarget
            starting = 'echo Starting SCons && '
            if cmdargs:
                cmdargs = ' ' + cmdargs
            else:
                cmdargs = ''
            buildcmd = xmlify(starting + self.env.subst('$MSVSBUILDCOM', 1) + cmdargs)
            rebuildcmd = xmlify(starting + self.env.subst('$MSVSREBUILDCOM', 1) + cmdargs)
            cleancmd = xmlify(starting + self.env.subst('$MSVSCLEANCOM', 1) + cmdargs)
            preprocdefs = xmlify(';'.join(processDefines(cppdefines)))
            includepath = xmlify(';'.join(processIncludes(cpppaths, self.env, None, None)))
            if not env_has_buildtarget:
                del self.env['MSVSBUILDTARGET']
            self.file.write(V10DSPCommandLine % locals())
        self.file.write('\t</PropertyGroup>\n')
        self.filtersabs = self.dspabs + '.filters'
        try:
            self.filters_file = open(self.filtersabs, 'w')
        except IOError as detail:
            raise SCons.Errors.InternalError('Unable to open "' + self.filtersabs + '" for writing:' + str(detail))
        self.filters_file.write('<?xml version="1.0" encoding="utf-8"?>\n<Project ToolsVersion="%s" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n' % self.versionstr)
        self.PrintSourceFiles()
        self.filters_file.write('</Project>')
        self.filters_file.close()
        self.file.write('\t<Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />\n\t<ImportGroup Label="ExtensionTargets">\n\t</ImportGroup>\n</Project>\n')
        if self.nokeep == 0:
            pdata = pickle.dumps(self.configs, PICKLE_PROTOCOL)
            pdata = base64.b64encode(pdata).decode()
            self.file.write('<!-- SCons Data:\n' + pdata + '\n')
            pdata = pickle.dumps(self.sources, PICKLE_PROTOCOL)
            pdata = base64.b64encode(pdata).decode()
            self.file.write(pdata + '-->\n')

    def printFilters(self, hierarchy, name):
        if False:
            for i in range(10):
                print('nop')
        sorteditems = sorted(hierarchy.items(), key=lambda a: a[0].lower())
        for (key, value) in sorteditems:
            if SCons.Util.is_Dict(value):
                filter_name = name + '\\' + key
                self.filters_file.write('\t\t<Filter Include="%s">\n\t\t\t<UniqueIdentifier>%s</UniqueIdentifier>\n\t\t</Filter>\n' % (filter_name, _generateGUID(self.dspabs, filter_name)))
                self.printFilters(value, filter_name)

    def printSources(self, hierarchy, kind, commonprefix, filter_name):
        if False:
            for i in range(10):
                print('nop')
        keywords = {'Source Files': 'ClCompile', 'Header Files': 'ClInclude', 'Local Headers': 'ClInclude', 'Resource Files': 'None', 'Other Files': 'None'}
        sorteditems = sorted(hierarchy.items(), key=lambda a: a[0].lower())
        for (key, value) in sorteditems:
            if SCons.Util.is_Dict(value):
                self.printSources(value, kind, commonprefix, filter_name + '\\' + key)
        for (key, value) in sorteditems:
            if SCons.Util.is_String(value):
                file = value
                if commonprefix:
                    file = os.path.join(commonprefix, value)
                file = os.path.normpath(file)
                self.file.write('\t\t<%s Include="%s" />\n' % (keywords[kind], file))
                self.filters_file.write('\t\t<%s Include="%s">\n\t\t\t<Filter>%s</Filter>\n\t\t</%s>\n' % (keywords[kind], file, filter_name, keywords[kind]))

    def PrintSourceFiles(self):
        if False:
            print('Hello World!')
        categories = {'Source Files': 'cpp;c;cxx;l;y;def;odl;idl;hpj;bat', 'Header Files': 'h;hpp;hxx;hm;inl', 'Local Headers': 'h;hpp;hxx;hm;inl', 'Resource Files': 'r;rc;ico;cur;bmp;dlg;rc2;rct;bin;cnt;rtf;gif;jpg;jpeg;jpe', 'Other Files': ''}
        cats = sorted([k for k in list(categories.keys()) if self.sources[k]], key=lambda a: a.lower())
        self.filters_file.write('\t<ItemGroup>\n')
        for kind in cats:
            self.filters_file.write('\t\t<Filter Include="%s">\n\t\t\t<UniqueIdentifier>{7b42d31d-d53c-4868-8b92-ca2bc9fc052f}</UniqueIdentifier>\n\t\t\t<Extensions>%s</Extensions>\n\t\t</Filter>\n' % (kind, categories[kind]))
            sources = self.sources[kind]
            commonprefix = None
            s = list(map(os.path.normpath, sources))
            cp = os.path.dirname(os.path.commonprefix(s))
            if cp and s[0][len(cp)] == os.sep:
                sources = [s[len(cp) + 1:] for s in sources]
                commonprefix = cp
            hierarchy = makeHierarchy(sources)
            self.printFilters(hierarchy, kind)
        self.filters_file.write('\t</ItemGroup>\n')
        for kind in cats:
            self.file.write('\t<ItemGroup>\n')
            self.filters_file.write('\t<ItemGroup>\n')
            sources = self.sources[kind]
            commonprefix = None
            s = list(map(os.path.normpath, sources))
            cp = os.path.dirname(os.path.commonprefix(s))
            if cp and s[0][len(cp)] == os.sep:
                sources = [s[len(cp) + 1:] for s in sources]
                commonprefix = cp
            hierarchy = makeHierarchy(sources)
            self.printSources(hierarchy, kind, commonprefix, kind)
            self.file.write('\t</ItemGroup>\n')
            self.filters_file.write('\t</ItemGroup>\n')
        self.file.write('\t<ItemGroup>\n\t\t<None Include="%s" />\n\t</ItemGroup>\n' % str(self.sconscript))

    def Parse(self):
        if False:
            for i in range(10):
                print('nop')
        print('_GenerateV10DSP.Parse()')

    def Build(self):
        if False:
            while True:
                i = 10
        try:
            self.file = open(self.dspabs, 'w')
        except IOError as detail:
            raise SCons.Errors.InternalError('Unable to open "' + self.dspabs + '" for writing:' + str(detail))
        else:
            self.PrintHeader()
            self.PrintProject()
            self.file.close()
        _GenerateV10User.Build(self)

class _DSWGenerator(object):
    """ Base class for DSW generators """

    def __init__(self, dswfile, source, env):
        if False:
            print('Hello World!')
        self.dswfile = os.path.normpath(str(dswfile))
        self.dsw_folder_path = os.path.dirname(os.path.abspath(self.dswfile))
        self.env = env
        if 'projects' not in env:
            raise SCons.Errors.UserError("You must specify a 'projects' argument to create an MSVSSolution.")
        projects = env['projects']
        if not SCons.Util.is_List(projects):
            raise SCons.Errors.InternalError("The 'projects' argument must be a list of nodes.")
        projects = SCons.Util.flatten(projects)
        if len(projects) < 1:
            raise SCons.Errors.UserError('You must specify at least one project to create an MSVSSolution.')
        self.dspfiles = list(map(str, projects))
        if 'name' in self.env:
            self.name = self.env['name']
        else:
            self.name = os.path.basename(SCons.Util.splitext(self.dswfile)[0])
        self.name = self.env.subst(self.name)

    def Build(self):
        if False:
            i = 10
            return i + 15
        pass

class _GenerateV7DSW(_DSWGenerator):
    """Generates a Solution file for MSVS .NET"""

    def __init__(self, dswfile, source, env):
        if False:
            i = 10
            return i + 15
        _DSWGenerator.__init__(self, dswfile, source, env)
        self.file = None
        self.version = self.env['MSVS_VERSION']
        (self.version_num, self.suite) = msvs_parse_version(self.version)
        self.versionstr = '7.00'
        if self.version_num >= 11.0:
            self.versionstr = '12.00'
        elif self.version_num >= 10.0:
            self.versionstr = '11.00'
        elif self.version_num >= 9.0:
            self.versionstr = '10.00'
        elif self.version_num >= 8.0:
            self.versionstr = '9.00'
        elif self.version_num >= 7.1:
            self.versionstr = '8.00'
        if 'slnguid' in env and env['slnguid']:
            self.slnguid = env['slnguid']
        else:
            self.slnguid = _generateGUID(dswfile, self.name)
        self.configs = {}
        self.nokeep = 0
        if 'nokeep' in env and env['variant'] != 0:
            self.nokeep = 1
        if self.nokeep == 0 and os.path.exists(self.dswfile):
            self.Parse()

        def AddConfig(self, variant, dswfile=dswfile):
            if False:
                while True:
                    i = 10
            config = Config()
            match = re.match('(.*)\\|(.*)', variant)
            if match:
                config.variant = match.group(1)
                config.platform = match.group(2)
            else:
                config.variant = variant
                config.platform = 'Win32'
            self.configs[variant] = config
            print("Adding '" + self.name + ' - ' + config.variant + '|' + config.platform + "' to '" + str(dswfile) + "'")
        if 'variant' not in env:
            raise SCons.Errors.InternalError("You must specify a 'variant' argument (i.e. 'Debug' or " + "'Release') to create an MSVS Solution File.")
        elif SCons.Util.is_String(env['variant']):
            AddConfig(self, env['variant'])
        elif SCons.Util.is_List(env['variant']):
            for variant in env['variant']:
                AddConfig(self, variant)
        self.platforms = []
        for key in list(self.configs.keys()):
            platform = self.configs[key].platform
            if platform not in self.platforms:
                self.platforms.append(platform)

        def GenerateProjectFilesInfo(self):
            if False:
                i = 10
                return i + 15
            for dspfile in self.dspfiles:
                (dsp_folder_path, name) = os.path.split(dspfile)
                dsp_folder_path = os.path.abspath(dsp_folder_path)
                if SCons.Util.splitext(name)[1] == '.filters':
                    continue
                dsp_relative_folder_path = os.path.relpath(dsp_folder_path, self.dsw_folder_path)
                if dsp_relative_folder_path == os.curdir:
                    dsp_relative_file_path = name
                else:
                    dsp_relative_file_path = os.path.join(dsp_relative_folder_path, name)
                dspfile_info = {'NAME': name, 'GUID': _generateGUID(dspfile, ''), 'FOLDER_PATH': dsp_folder_path, 'FILE_PATH': dspfile, 'SLN_RELATIVE_FOLDER_PATH': dsp_relative_folder_path, 'SLN_RELATIVE_FILE_PATH': dsp_relative_file_path}
                self.dspfiles_info.append(dspfile_info)
        self.dspfiles_info = []
        GenerateProjectFilesInfo(self)

    def Parse(self):
        if False:
            print('Hello World!')
        try:
            dswfile = open(self.dswfile, 'r')
        except IOError:
            return
        line = dswfile.readline()
        while line:
            if line[:9] == 'EndGlobal':
                break
            line = dswfile.readline()
        line = dswfile.readline()
        datas = line
        while line:
            line = dswfile.readline()
            datas = datas + line
        dswfile.close()
        try:
            datas = base64.decodestring(datas)
            data = pickle.loads(datas)
        except KeyboardInterrupt:
            raise
        except:
            return
        self.configs.update(data)

    def PrintSolution(self):
        if False:
            while True:
                i = 10
        'Writes a solution file'
        self.file.write('Microsoft Visual Studio Solution File, Format Version %s\n' % self.versionstr)
        if self.version_num >= 14.2:
            self.file.write('# Visual Studio 16\n')
        elif self.version_num > 14.0:
            self.file.write('# Visual Studio 15\n')
        elif self.version_num >= 12.0:
            self.file.write('# Visual Studio 14\n')
        elif self.version_num >= 11.0:
            self.file.write('# Visual Studio 11\n')
        elif self.version_num >= 10.0:
            self.file.write('# Visual Studio 2010\n')
        elif self.version_num >= 9.0:
            self.file.write('# Visual Studio 2008\n')
        elif self.version_num >= 8.0:
            self.file.write('# Visual Studio 2005\n')
        for dspinfo in self.dspfiles_info:
            name = dspinfo['NAME']
            (base, suffix) = SCons.Util.splitext(name)
            if suffix == '.vcproj':
                name = base
            self.file.write('Project("%s") = "%s", "%s", "%s"\n' % (external_makefile_guid, name, dspinfo['SLN_RELATIVE_FILE_PATH'], dspinfo['GUID']))
            if 7.1 <= self.version_num < 8.0:
                self.file.write('\tProjectSection(ProjectDependencies) = postProject\n\tEndProjectSection\n')
            self.file.write('EndProject\n')
        self.file.write('Global\n')
        env = self.env
        if 'MSVS_SCC_PROVIDER' in env:
            scc_number_of_projects = len(self.dspfiles) + 1
            slnguid = self.slnguid
            scc_provider = env.get('MSVS_SCC_PROVIDER', '').replace(' ', '\\u0020')
            scc_project_name = env.get('MSVS_SCC_PROJECT_NAME', '').replace(' ', '\\u0020')
            scc_connection_root = env.get('MSVS_SCC_CONNECTION_ROOT', os.curdir)
            scc_local_path = os.path.relpath(scc_connection_root, self.dsw_folder_path).replace('\\', '\\\\')
            self.file.write('\tGlobalSection(SourceCodeControl) = preSolution\n\t\tSccNumberOfProjects = %(scc_number_of_projects)d\n\t\tSccProjectName0 = %(scc_project_name)s\n\t\tSccLocalPath0 = %(scc_local_path)s\n\t\tSccProvider0 = %(scc_provider)s\n\t\tCanCheckoutShared = true\n' % locals())
            sln_relative_path_from_scc = os.path.relpath(self.dsw_folder_path, scc_connection_root)
            if sln_relative_path_from_scc != os.curdir:
                self.file.write('\t\tSccProjectFilePathRelativizedFromConnection0 = %s\\\\\n' % sln_relative_path_from_scc.replace('\\', '\\\\'))
            if self.version_num < 8.0:
                self.file.write('\t\tSolutionUniqueID = %s\n' % slnguid)
            for dspinfo in self.dspfiles_info:
                i = self.dspfiles_info.index(dspinfo) + 1
                dsp_relative_file_path = dspinfo['SLN_RELATIVE_FILE_PATH'].replace('\\', '\\\\')
                dsp_scc_relative_folder_path = os.path.relpath(dspinfo['FOLDER_PATH'], scc_connection_root).replace('\\', '\\\\')
                self.file.write('\t\tSccProjectUniqueName%(i)s = %(dsp_relative_file_path)s\n\t\tSccLocalPath%(i)d = %(scc_local_path)s\n\t\tCanCheckoutShared = true\n\t\tSccProjectFilePathRelativizedFromConnection%(i)s = %(dsp_scc_relative_folder_path)s\\\\\n' % locals())
            self.file.write('\tEndGlobalSection\n')
        if self.version_num >= 8.0:
            self.file.write('\tGlobalSection(SolutionConfigurationPlatforms) = preSolution\n')
        else:
            self.file.write('\tGlobalSection(SolutionConfiguration) = preSolution\n')
        confkeys = sorted(self.configs.keys())
        cnt = 0
        for name in confkeys:
            variant = self.configs[name].variant
            platform = self.configs[name].platform
            if self.version_num >= 8.0:
                self.file.write('\t\t%s|%s = %s|%s\n' % (variant, platform, variant, platform))
            else:
                self.file.write('\t\tConfigName.%d = %s\n' % (cnt, variant))
            cnt = cnt + 1
        self.file.write('\tEndGlobalSection\n')
        if self.version_num <= 7.1:
            self.file.write('\tGlobalSection(ProjectDependencies) = postSolution\n\tEndGlobalSection\n')
        if self.version_num >= 8.0:
            self.file.write('\tGlobalSection(ProjectConfigurationPlatforms) = postSolution\n')
        else:
            self.file.write('\tGlobalSection(ProjectConfiguration) = postSolution\n')
        for name in confkeys:
            variant = self.configs[name].variant
            platform = self.configs[name].platform
            if self.version_num >= 8.0:
                for dspinfo in self.dspfiles_info:
                    guid = dspinfo['GUID']
                    self.file.write('\t\t%s.%s|%s.ActiveCfg = %s|%s\n\t\t%s.%s|%s.Build.0 = %s|%s\n' % (guid, variant, platform, variant, platform, guid, variant, platform, variant, platform))
            else:
                for dspinfo in self.dspfiles_info:
                    guid = dspinfo['GUID']
                    self.file.write('\t\t%s.%s.ActiveCfg = %s|%s\n\t\t%s.%s.Build.0 = %s|%s\n' % (guid, variant, variant, platform, guid, variant, variant, platform))
        self.file.write('\tEndGlobalSection\n')
        if self.version_num >= 8.0:
            self.file.write('\tGlobalSection(SolutionProperties) = preSolution\n\t\tHideSolutionNode = FALSE\n\tEndGlobalSection\n')
        else:
            self.file.write('\tGlobalSection(ExtensibilityGlobals) = postSolution\n\tEndGlobalSection\n\tGlobalSection(ExtensibilityAddIns) = postSolution\n\tEndGlobalSection\n')
        self.file.write('EndGlobal\n')
        if self.nokeep == 0:
            pdata = pickle.dumps(self.configs, PICKLE_PROTOCOL)
            pdata = base64.b64encode(pdata).decode()
            self.file.write(pdata)
            self.file.write('\n')

    def Build(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.file = open(self.dswfile, 'w')
        except IOError as detail:
            raise SCons.Errors.InternalError('Unable to open "' + self.dswfile + '" for writing:' + str(detail))
        else:
            self.PrintSolution()
            self.file.close()
V6DSWHeader = 'Microsoft Developer Studio Workspace File, Format Version 6.00\n# WARNING: DO NOT EDIT OR DELETE THIS WORKSPACE FILE!\n\n###############################################################################\n\nProject: "%(name)s"="%(dspfile)s" - Package Owner=<4>\n\nPackage=<5>\n{{{\n}}}\n\nPackage=<4>\n{{{\n}}}\n\n###############################################################################\n\nGlobal:\n\nPackage=<5>\n{{{\n}}}\n\nPackage=<3>\n{{{\n}}}\n\n###############################################################################\n'

class _GenerateV6DSW(_DSWGenerator):
    """Generates a Workspace file for MSVS 6.0"""

    def PrintWorkspace(self):
        if False:
            print('Hello World!')
        ' writes a DSW file '
        name = self.name
        dspfile = os.path.relpath(self.dspfiles[0], self.dsw_folder_path)
        self.file.write(V6DSWHeader % locals())

    def Build(self):
        if False:
            while True:
                i = 10
        try:
            self.file = open(self.dswfile, 'w')
        except IOError as detail:
            raise SCons.Errors.InternalError('Unable to open "' + self.dswfile + '" for writing:' + str(detail))
        else:
            self.PrintWorkspace()
            self.file.close()

def GenerateDSP(dspfile, source, env):
    if False:
        for i in range(10):
            print('nop')
    'Generates a Project file based on the version of MSVS that is being used'
    version_num = 6.0
    if 'MSVS_VERSION' in env:
        (version_num, suite) = msvs_parse_version(env['MSVS_VERSION'])
    if version_num >= 10.0:
        g = _GenerateV10DSP(dspfile, source, env)
        g.Build()
    elif version_num >= 7.0:
        g = _GenerateV7DSP(dspfile, source, env)
        g.Build()
    else:
        g = _GenerateV6DSP(dspfile, source, env)
        g.Build()

def GenerateDSW(dswfile, source, env):
    if False:
        for i in range(10):
            print('nop')
    'Generates a Solution/Workspace file based on the version of MSVS that is being used'
    version_num = 6.0
    if 'MSVS_VERSION' in env:
        (version_num, suite) = msvs_parse_version(env['MSVS_VERSION'])
    if version_num >= 7.0:
        g = _GenerateV7DSW(dswfile, source, env)
        g.Build()
    else:
        g = _GenerateV6DSW(dswfile, source, env)
        g.Build()

def GetMSVSProjectSuffix(target, source, env, for_signature):
    if False:
        return 10
    return env['MSVS']['PROJECTSUFFIX']

def GetMSVSSolutionSuffix(target, source, env, for_signature):
    if False:
        print('Hello World!')
    return env['MSVS']['SOLUTIONSUFFIX']

def GenerateProject(target, source, env):
    if False:
        i = 10
        return i + 15
    builddspfile = target[0]
    dspfile = builddspfile.srcnode()
    if dspfile is not builddspfile:
        try:
            bdsp = open(str(builddspfile), 'w+')
        except IOError as detail:
            print('Unable to open "' + str(dspfile) + '" for writing:', detail, '\n')
            raise
        bdsp.write('This is just a placeholder file.\nThe real project file is here:\n%s\n' % dspfile.get_abspath())
        bdsp.close()
    GenerateDSP(dspfile, source, env)
    if env.get('auto_build_solution', 1):
        builddswfile = target[1]
        dswfile = builddswfile.srcnode()
        if dswfile is not builddswfile:
            try:
                bdsw = open(str(builddswfile), 'w+')
            except IOError as detail:
                print('Unable to open "' + str(dspfile) + '" for writing:', detail, '\n')
                raise
            bdsw.write('This is just a placeholder file.\nThe real workspace file is here:\n%s\n' % dswfile.get_abspath())
            bdsw.close()
        GenerateDSW(dswfile, source, env)

def GenerateSolution(target, source, env):
    if False:
        print('Hello World!')
    GenerateDSW(target[0], source, env)

def projectEmitter(target, source, env):
    if False:
        i = 10
        return i + 15
    'Sets up the DSP dependencies.'
    if source[0] == target[0]:
        source = []
    (base, suff) = SCons.Util.splitext(str(target[0]))
    suff = env.subst('$MSVSPROJECTSUFFIX')
    target[0] = base + suff
    if not source:
        source = 'prj_inputs:'
        source = source + env.subst('$MSVSSCONSCOM', 1)
        source = source + env.subst('$MSVSENCODING', 1)
        preprocdefs = xmlify(';'.join(processDefines(env.get('CPPDEFINES', []))))
        includepath = xmlify(';'.join(processIncludes(env.get('CPPPATH', []), env, None, None)))
        source = source + '; ppdefs:%s incpath:%s' % (preprocdefs, includepath)
        if 'buildtarget' in env and env['buildtarget'] is not None:
            if SCons.Util.is_String(env['buildtarget']):
                source = source + ' "%s"' % env['buildtarget']
            elif SCons.Util.is_List(env['buildtarget']):
                for bt in env['buildtarget']:
                    if SCons.Util.is_String(bt):
                        source = source + ' "%s"' % bt
                    else:
                        try:
                            source = source + ' "%s"' % bt.get_abspath()
                        except AttributeError:
                            raise SCons.Errors.InternalError('buildtarget can be a string, a node, a list of strings or nodes, or None')
            else:
                try:
                    source = source + ' "%s"' % env['buildtarget'].get_abspath()
                except AttributeError:
                    raise SCons.Errors.InternalError('buildtarget can be a string, a node, a list of strings or nodes, or None')
        if 'outdir' in env and env['outdir'] is not None:
            if SCons.Util.is_String(env['outdir']):
                source = source + ' "%s"' % env['outdir']
            elif SCons.Util.is_List(env['outdir']):
                for s in env['outdir']:
                    if SCons.Util.is_String(s):
                        source = source + ' "%s"' % s
                    else:
                        try:
                            source = source + ' "%s"' % s.get_abspath()
                        except AttributeError:
                            raise SCons.Errors.InternalError('outdir can be a string, a node, a list of strings or nodes, or None')
            else:
                try:
                    source = source + ' "%s"' % env['outdir'].get_abspath()
                except AttributeError:
                    raise SCons.Errors.InternalError('outdir can be a string, a node, a list of strings or nodes, or None')
        if 'name' in env:
            if SCons.Util.is_String(env['name']):
                source = source + ' "%s"' % env['name']
            else:
                raise SCons.Errors.InternalError('name must be a string')
        if 'variant' in env:
            if SCons.Util.is_String(env['variant']):
                source = source + ' "%s"' % env['variant']
            elif SCons.Util.is_List(env['variant']):
                for variant in env['variant']:
                    if SCons.Util.is_String(variant):
                        source = source + ' "%s"' % variant
                    else:
                        raise SCons.Errors.InternalError('name must be a string or a list of strings')
            else:
                raise SCons.Errors.InternalError('variant must be a string or a list of strings')
        else:
            raise SCons.Errors.InternalError('variant must be specified')
        for s in _DSPGenerator.srcargs:
            if s in env:
                if SCons.Util.is_String(env[s]):
                    source = source + ' "%s' % env[s]
                elif SCons.Util.is_List(env[s]):
                    for t in env[s]:
                        if SCons.Util.is_String(t):
                            source = source + ' "%s"' % t
                        else:
                            raise SCons.Errors.InternalError(s + ' must be a string or a list of strings')
                else:
                    raise SCons.Errors.InternalError(s + ' must be a string or a list of strings')
        source = source + ' "%s"' % str(target[0])
        source = [SCons.Node.Python.Value(source)]
    targetlist = [target[0]]
    sourcelist = source
    if env.get('auto_build_solution', 1):
        env['projects'] = [env.File(t).srcnode() for t in targetlist]
        (t, s) = solutionEmitter(target, target, env)
        targetlist = targetlist + t
    version_num = 6.0
    if 'MSVS_VERSION' in env:
        (version_num, suite) = msvs_parse_version(env['MSVS_VERSION'])
    if version_num >= 10.0:
        targetlist.append(targetlist[0] + '.filters')
    return (targetlist, sourcelist)

def solutionEmitter(target, source, env):
    if False:
        print('Hello World!')
    'Sets up the DSW dependencies.'
    if source[0] == target[0]:
        source = []
    (base, suff) = SCons.Util.splitext(str(target[0]))
    suff = env.subst('$MSVSSOLUTIONSUFFIX')
    target[0] = base + suff
    if not source:
        source = 'sln_inputs:'
        if 'name' in env:
            if SCons.Util.is_String(env['name']):
                source = source + ' "%s"' % env['name']
            else:
                raise SCons.Errors.InternalError('name must be a string')
        if 'variant' in env:
            if SCons.Util.is_String(env['variant']):
                source = source + ' "%s"' % env['variant']
            elif SCons.Util.is_List(env['variant']):
                for variant in env['variant']:
                    if SCons.Util.is_String(variant):
                        source = source + ' "%s"' % variant
                    else:
                        raise SCons.Errors.InternalError('name must be a string or a list of strings')
            else:
                raise SCons.Errors.InternalError('variant must be a string or a list of strings')
        else:
            raise SCons.Errors.InternalError('variant must be specified')
        if 'slnguid' in env:
            if SCons.Util.is_String(env['slnguid']):
                source = source + ' "%s"' % env['slnguid']
            else:
                raise SCons.Errors.InternalError('slnguid must be a string')
        if 'projects' in env:
            if SCons.Util.is_String(env['projects']):
                source = source + ' "%s"' % env['projects']
            elif SCons.Util.is_List(env['projects']):
                for t in env['projects']:
                    if SCons.Util.is_String(t):
                        source = source + ' "%s"' % t
        source = source + ' "%s"' % str(target[0])
        source = [SCons.Node.Python.Value(source)]
    return ([target[0]], source)
projectAction = SCons.Action.Action(GenerateProject, None)
solutionAction = SCons.Action.Action(GenerateSolution, None)
projectBuilder = SCons.Builder.Builder(action='$MSVSPROJECTCOM', suffix='$MSVSPROJECTSUFFIX', emitter=projectEmitter)
solutionBuilder = SCons.Builder.Builder(action='$MSVSSOLUTIONCOM', suffix='$MSVSSOLUTIONSUFFIX', emitter=solutionEmitter)
default_MSVS_SConscript = None

def generate(env):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for Microsoft Visual\n    Studio project files to an Environment.'
    try:
        env['BUILDERS']['MSVSProject']
    except KeyError:
        env['BUILDERS']['MSVSProject'] = projectBuilder
    try:
        env['BUILDERS']['MSVSSolution']
    except KeyError:
        env['BUILDERS']['MSVSSolution'] = solutionBuilder
    env['MSVSPROJECTCOM'] = projectAction
    env['MSVSSOLUTIONCOM'] = solutionAction
    if SCons.Script.call_stack:
        env['MSVSSCONSCRIPT'] = SCons.Script.call_stack[0].sconscript
    else:
        global default_MSVS_SConscript
        if default_MSVS_SConscript is None:
            default_MSVS_SConscript = env.File('SConstruct')
        env['MSVSSCONSCRIPT'] = default_MSVS_SConscript
    if 'MSVSSCONS' not in env:
        env['MSVSSCONS'] = '"%s" -c "%s"' % (python_executable, getExecScriptMain(env))
    if 'MSVSSCONSFLAGS' not in env:
        env['MSVSSCONSFLAGS'] = '-C "${MSVSSCONSCRIPT.dir.get_abspath()}" -f ${MSVSSCONSCRIPT.name}'
    env['MSVSSCONSCOM'] = '$MSVSSCONS $MSVSSCONSFLAGS'
    env['MSVSBUILDCOM'] = '$MSVSSCONSCOM "$MSVSBUILDTARGET"'
    env['MSVSREBUILDCOM'] = '$MSVSSCONSCOM "$MSVSBUILDTARGET"'
    env['MSVSCLEANCOM'] = '$MSVSSCONSCOM -c "$MSVSBUILDTARGET"'
    msvc_setup_env_once(env)
    if 'MSVS_VERSION' in env:
        (version_num, suite) = msvs_parse_version(env['MSVS_VERSION'])
    else:
        (version_num, suite) = (7.0, None)
    if 'MSVS' not in env:
        env['MSVS'] = {}
    if version_num < 7.0:
        env['MSVS']['PROJECTSUFFIX'] = '.dsp'
        env['MSVS']['SOLUTIONSUFFIX'] = '.dsw'
    elif version_num < 10.0:
        env['MSVS']['PROJECTSUFFIX'] = '.vcproj'
        env['MSVS']['SOLUTIONSUFFIX'] = '.sln'
    else:
        env['MSVS']['PROJECTSUFFIX'] = '.vcxproj'
        env['MSVS']['SOLUTIONSUFFIX'] = '.sln'
    if version_num >= 10.0:
        env['MSVSENCODING'] = 'utf-8'
    else:
        env['MSVSENCODING'] = 'Windows-1252'
    env['GET_MSVSPROJECTSUFFIX'] = GetMSVSProjectSuffix
    env['GET_MSVSSOLUTIONSUFFIX'] = GetMSVSSolutionSuffix
    env['MSVSPROJECTSUFFIX'] = '${GET_MSVSPROJECTSUFFIX}'
    env['MSVSSOLUTIONSUFFIX'] = '${GET_MSVSSOLUTIONSUFFIX}'
    env['SCONS_HOME'] = os.environ.get('SCONS_HOME')

def exists(env):
    if False:
        return 10
    return msvc_exists(env)