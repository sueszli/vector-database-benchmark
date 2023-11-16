"""SCons.Tool.Packaging.ipk
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import os
import SCons.Builder
import SCons.Node.FS
import SCons.Util
from SCons.Tool.packaging import stripinstallbuilder, putintopackageroot

def package(env, target, source, PACKAGEROOT, NAME, VERSION, DESCRIPTION, SUMMARY, X_IPK_PRIORITY, X_IPK_SECTION, SOURCE_URL, X_IPK_MAINTAINER, X_IPK_DEPENDS, **kw):
    if False:
        i = 10
        return i + 15
    ' This function prepares the packageroot directory for packaging with the\n    ipkg builder.\n    '
    SCons.Tool.Tool('ipkg').generate(env)
    bld = env['BUILDERS']['Ipkg']
    (target, source) = stripinstallbuilder(target, source, env)
    (target, source) = putintopackageroot(target, source, env, PACKAGEROOT)
    archmap = {'i686': 'i386', 'i586': 'i386', 'i486': 'i386'}
    buildarchitecture = os.uname()[4]
    buildarchitecture = archmap.get(buildarchitecture, buildarchitecture)
    if 'ARCHITECTURE' in kw:
        buildarchitecture = kw['ARCHITECTURE']
    loc = locals()
    del loc['kw']
    kw.update(loc)
    del kw['source'], kw['target'], kw['env']
    specfile = gen_ipk_dir(PACKAGEROOT, source, env, kw)
    if str(target[0]) == '%s-%s' % (NAME, VERSION):
        target = ['%s_%s_%s.ipk' % (NAME, VERSION, buildarchitecture)]
    return bld(env, target, specfile, **kw)

def gen_ipk_dir(proot, source, env, kw):
    if False:
        while True:
            i = 10
    if SCons.Util.is_String(proot):
        proot = env.Dir(proot)
    s_bld = SCons.Builder.Builder(action=build_specfiles)
    spec_target = []
    control = proot.Dir('CONTROL')
    spec_target.append(control.File('control'))
    spec_target.append(control.File('conffiles'))
    spec_target.append(control.File('postrm'))
    spec_target.append(control.File('prerm'))
    spec_target.append(control.File('postinst'))
    spec_target.append(control.File('preinst'))
    s_bld(env, spec_target, source, **kw)
    return proot

def build_specfiles(source, target, env):
    if False:
        while True:
            i = 10
    ' Filter the targets for the needed files and use the variables in env\n    to create the specfile.\n    '
    opened_files = {}

    def open_file(needle, haystack=None):
        if False:
            while True:
                i = 10
        try:
            return opened_files[needle]
        except KeyError:
            files = filter(lambda x: x.get_path().rfind(needle) != -1, haystack)
            file = list(files)[0]
            opened_files[needle] = open(file.get_abspath(), 'w')
            return opened_files[needle]
    control_file = open_file('control', target)
    if 'X_IPK_DESCRIPTION' not in env:
        env['X_IPK_DESCRIPTION'] = '%s\n %s' % (env['SUMMARY'], env['DESCRIPTION'].replace('\n', '\n '))
    content = '\nPackage: $NAME\nVersion: $VERSION\nPriority: $X_IPK_PRIORITY\nSection: $X_IPK_SECTION\nSource: $SOURCE_URL\nArchitecture: $ARCHITECTURE\nMaintainer: $X_IPK_MAINTAINER\nDepends: $X_IPK_DEPENDS\nDescription: $X_IPK_DESCRIPTION\n'
    control_file.write(env.subst(content))
    for f in [x for x in source if 'PACKAGING_CONFIG' in dir(x)]:
        config = open_file('conffiles')
        config.write(f.PACKAGING_INSTALL_LOCATION)
        config.write('\n')
    for str in 'POSTRM PRERM POSTINST PREINST'.split():
        name = 'PACKAGING_X_IPK_%s' % str
        for f in [x for x in source if name in dir(x)]:
            file = open_file(name)
            file.write(env[str])
    for f in opened_files.values():
        f.close()
    if 'CHANGE_SPECFILE' in env:
        content += env['CHANGE_SPECFILE'](target)
    return 0