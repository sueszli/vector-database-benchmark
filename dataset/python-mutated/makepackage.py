import sys
import os
import shutil
import glob
import re
import subprocess
from makepandacore import *
from installpanda import *
INSTALLER_DEB_FILE = "\nPackage: panda3dMAJOR\nVersion: VERSION\nSection: libdevel\nPriority: optional\nArchitecture: ARCH\nEssential: no\nDepends: DEPENDS\nRecommends: RECOMMENDS\nProvides: PROVIDES\nConflicts: PROVIDES\nReplaces: PROVIDES\nMaintainer: rdb <me@rdb.name>\nInstalled-Size: INSTSIZE\nDescription: Panda3D free 3D engine SDK\n Panda3D is a game engine which includes graphics, audio, I/O, collision detection, and other abilities relevant to the creation of 3D games. Panda3D is open source and free software under the revised BSD license, and can be used for both free and commercial game development at no financial cost.\n Panda3D's intended game-development language is Python. The engine itself is written in C++, and utilizes an automatic wrapper-generator to expose the complete functionality of the engine in a Python interface.\n .\n This package contains the SDK for development with Panda3D.\n\n"
INSTALLER_SPEC_FILE = "\nSummary: The Panda3D free 3D engine SDK\nName: panda3d\nVersion: VERSION\nRelease: RPMRELEASE\nLicense: BSD License\nGroup: Development/Libraries\nBuildRoot: PANDASOURCE/targetroot\n%description\nPanda3D is a game engine which includes graphics, audio, I/O, collision detection, and other abilities relevant to the creation of 3D games. Panda3D is open source and free software under the revised BSD license, and can be used for both free and commercial game development at no financial cost.\nPanda3D's intended game-development language is Python. The engine itself is written in C++, and utilizes an automatic wrapper-generator to expose the complete functionality of the engine in a Python interface.\n\nThis package contains the SDK for development with Panda3D.\n%post\n/sbin/ldconfig\n%postun\n/sbin/ldconfig\n%files\n%defattr(-,root,root)\n/etc/Confauto.prc\n/etc/Config.prc\n/usr/share/panda3d\n/etc/ld.so.conf.d/panda3d.conf\n/usr/%_lib/panda3d\n/usr/include/panda3d\n"
INSTALLER_SPEC_FILE_PVIEW = '/usr/share/applications/pview.desktop\n/usr/share/mime-info/panda3d.mime\n/usr/share/mime-info/panda3d.keys\n/usr/share/mime/packages/panda3d.xml\n/usr/share/application-registry/panda3d.applications\n'
Info_plist = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0">\n<dict>\n  <key>CFBundleIdentifier</key>\n  <string>{package_id}</string>\n  <key>CFBundleShortVersionString</key>\n  <string>{version}</string>\n  <key>IFPkgFlagRelocatable</key>\n  <false/>\n  <key>IFPkgFlagAuthorizationAction</key>\n  <string>RootAuthorization</string>\n  <key>IFPkgFlagAllowBackRev</key>\n  <true/>\n</dict>\n</plist>\n'
INSTALLER_PKG_DESCR_FILE = "\nPanda3D is a game engine which includes graphics, audio, I/O, collision detection, and other abilities relevant to the creation of 3D games. Panda3D is open source and free software under the revised BSD license, and can be used for both free and commercial game development at no financial cost.\nPanda3D's intended game-development language is Python. The engine itself is written in C++, and utilizes an automatic wrapper-generator to expose the complete functionality of the engine in a Python interface.\n\nThis package contains the SDK for development with Panda3D.\n\nWWW: https://www.panda3d.org/\n"
INSTALLER_PKG_MANIFEST_FILE = '\nname: NAME\nversion: VERSION\narch: ARCH\norigin: ORIGIN\ncomment: "Panda3D free 3D engine SDK"\nwww: https://www.panda3d.org\nmaintainer: rdb <me@rdb.name>\nprefix: /usr/local\nflatsize: INSTSIZEMB\ndeps: {DEPENDS}\n'
MACOS_SCRIPT_PREFIX = '#!/bin/bash\nIFS=.\nread -a version_info <<< "`sw_vers -productVersion`"\nif (( ${version_info[0]} == 10 && ${version_info[1]} < 15 )); then\n'
MACOS_SCRIPT_POSTFIX = 'fi\n'

def MakeInstallerNSIS(version, file, title, installdir, compressor='lzma', **kwargs):
    if False:
        i = 10
        return i + 15
    outputdir = GetOutputDir()
    if os.path.isfile(file):
        os.remove(file)
    elif os.path.isdir(file):
        shutil.rmtree(file)
    if GetTargetArch() == 'x64':
        regview = '64'
    else:
        regview = '32'
    print('Building ' + title + ' installer at %s' % file)
    if compressor != 'lzma':
        print('Note: you are using zlib, which is faster, but lzma gives better compression.')
    if os.path.exists('nsis-output.exe'):
        os.remove('nsis-output.exe')
    WriteFile(outputdir + '/tmp/__init__.py', '')
    nsis_defs = {'COMPRESSOR': compressor, 'TITLE': title, 'INSTALLDIR': installdir, 'OUTFILE': '..\\' + file, 'BUILT': '..\\' + outputdir, 'SOURCE': '..', 'REGVIEW': regview, 'MAJOR_VER': '.'.join(version.split('.')[:2])}
    if os.path.isfile(os.path.join(outputdir, 'python', 'python.exe')):
        py_dlls = glob.glob(os.path.join(outputdir, 'python', 'python[0-9][0-9].dll')) + glob.glob(os.path.join(outputdir, 'python', 'python[0-9][0-9]_d.dll')) + glob.glob(os.path.join(outputdir, 'python', 'python[0-9][0-9][0-9].dll')) + glob.glob(os.path.join(outputdir, 'python', 'python[0-9][0-9][0-9]_d.dll'))
        assert py_dlls
        py_dll = os.path.basename(py_dlls[0])
        py_dllver = py_dll.strip('.DHLNOPTY_dhlnopty')
        pyver = py_dllver[0] + '.' + py_dllver[1:]
        if GetTargetArch() != 'x64':
            pyver += '-32'
        nsis_defs['INCLUDE_PYVER'] = pyver
    if GetHost() == 'windows':
        cmd = os.path.join(GetThirdpartyBase(), 'win-nsis', 'makensis') + ' /V2'
        for item in nsis_defs.items():
            cmd += ' /D%s="%s"' % item
    else:
        cmd = 'makensis -V2'
        for item in nsis_defs.items():
            cmd += ' -D%s="%s"' % item
    cmd += ' "makepanda\\installer.nsi"'
    oscmd(cmd)

def MakeDebugSymbolZipArchive(zipname):
    if False:
        i = 10
        return i + 15
    import zipfile
    outputdir = GetOutputDir()
    zip = zipfile.ZipFile(zipname + '.zip', 'w', zipfile.ZIP_DEFLATED)
    for fn in glob.glob(os.path.join(outputdir, 'bin', '*.pdb')):
        zip.write(fn, 'bin/' + os.path.basename(fn))
    for fn in glob.glob(os.path.join(outputdir, 'panda3d', '*.pdb')):
        zip.write(fn, 'panda3d/' + os.path.basename(fn))
    for fn in glob.glob(os.path.join(outputdir, 'plugins', '*.pdb')):
        zip.write(fn, 'plugins/' + os.path.basename(fn))
    for fn in glob.glob(os.path.join(outputdir, 'python', '*.pdb')):
        zip.write(fn, 'python/' + os.path.basename(fn))
    for fn in glob.glob(os.path.join(outputdir, 'python', 'DLLs', '*.pdb')):
        zip.write(fn, 'python/DLLs/' + os.path.basename(fn))
    zip.close()

def MakeDebugSymbolSevenZipArchive(zipname, compressor):
    if False:
        print('Hello World!')
    zipname += '.7z'
    flags = ['-t7z', '-y']
    if compressor == 'zlib':
        flags.extend(['-mx=3'])
    if os.path.exists(zipname):
        os.remove(zipname)
    outputdir = GetOutputDir()
    zipname = os.path.relpath(zipname, outputdir)
    cmd = [GetSevenZip(), 'a']
    cmd.extend(flags)
    cmd.extend(['-ir!*.pdb', '-x!' + os.path.join('tmp', '*'), zipname])
    subprocess.call(cmd, stdout=subprocess.DEVNULL, cwd=outputdir)

def MakeDebugSymbolArchive(zipname, compressor):
    if False:
        return 10
    if HasSevenZip():
        MakeDebugSymbolSevenZipArchive(zipname, compressor)
    else:
        MakeDebugSymbolZipArchive(zipname)

def MakeInstallerLinux(version, debversion=None, rpmversion=None, rpmrelease=1, python_versions=[], **kwargs):
    if False:
        while True:
            i = 10
    outputdir = GetOutputDir()
    install_python_versions = []
    for version_info in python_versions:
        if os.path.isdir('/usr/lib/python' + version_info['version']):
            install_python_versions.append(version_info)
    major_version = '.'.join(version.split('.')[:2])
    if not debversion:
        debversion = version
    if not rpmversion:
        rpmversion = version
    oscmd('rm -rf targetroot data.tar.gz control.tar.gz panda3d.spec')
    oscmd('mkdir -m 0755 targetroot')
    dpkg_present = False
    if os.path.exists('/usr/bin/dpkg-architecture') and os.path.exists('/usr/bin/dpkg-deb'):
        dpkg_present = True
    rpmbuild_present = False
    if os.path.exists('/usr/bin/rpmbuild'):
        rpmbuild_present = True
    if dpkg_present and rpmbuild_present:
        Warn('both dpkg and rpmbuild present.')
    if dpkg_present:
        lib_dir = GetDebLibDir()
        InstallPanda(destdir='targetroot', prefix='/usr', outputdir=outputdir, libdir=lib_dir, python_versions=install_python_versions)
        oscmd('chmod -R 755 targetroot/usr/share/panda3d')
        oscmd('mkdir -m 0755 -p targetroot/usr/share/man/man1')
        oscmd('install -m 0644 doc/man/*.1 targetroot/usr/share/man/man1/')
        oscmd('dpkg --print-architecture > ' + outputdir + '/tmp/architecture.txt')
        pkg_arch = ReadFile(outputdir + '/tmp/architecture.txt').strip()
        txt = INSTALLER_DEB_FILE[1:]
        txt = txt.replace('VERSION', debversion).replace('ARCH', pkg_arch).replace('MAJOR', major_version)
        txt = txt.replace('INSTSIZE', str(GetDirectorySize('targetroot') // 1024))
        oscmd('mkdir -m 0755 -p targetroot/DEBIAN')
        oscmd('cd targetroot && (find usr -type f -exec md5sum {} ;) > DEBIAN/md5sums')
        oscmd('cd targetroot && (find etc -type f -exec md5sum {} ;) >> DEBIAN/md5sums')
        WriteFile('targetroot/DEBIAN/conffiles', '/etc/Config.prc\n')
        WriteFile('targetroot/DEBIAN/postinst', '#!/bin/sh\necho running ldconfig\nldconfig\n')
        oscmd('cp targetroot/DEBIAN/postinst targetroot/DEBIAN/postrm')
        pkg_version = debversion
        pkg_name = 'panda3d' + major_version
        lib_pattern = 'debian/%s/usr/%s/panda3d/*.so*' % (pkg_name, lib_dir)
        bin_pattern = 'debian/%s/usr/bin/*' % pkg_name
        oscmd('mkdir targetroot/debian')
        oscmd('ln -s .. targetroot/debian/' + pkg_name)
        WriteFile('targetroot/debian/control', '')
        dpkg_shlibdeps = 'dpkg-shlibdeps'
        if GetVerbose():
            dpkg_shlibdeps += ' -v'
        pkg_name = 'panda3d' + major_version
        pkg_dir = 'debian/panda3d' + major_version
        oscmd(f'cd targetroot && dpkg-gensymbols -q -ODEBIAN/symbols -v{pkg_version} -p{pkg_name} -e{lib_pattern}')
        oscmd(f'cd targetroot && LD_LIBRARY_PATH=usr/{lib_dir}/panda3d {dpkg_shlibdeps} -Tdebian/substvars_dep --ignore-missing-info -x{pkg_name} -xlibphysx-extras {lib_pattern}')
        oscmd(f'cd targetroot && LD_LIBRARY_PATH=usr/{lib_dir}/panda3d {dpkg_shlibdeps} -Tdebian/substvars_rec --ignore-missing-info -x{pkg_name} {bin_pattern}')
        depends = ReadFile('targetroot/debian/substvars_dep').replace('shlibs:Depends=', '').strip()
        recommends = ReadFile('targetroot/debian/substvars_rec').replace('shlibs:Depends=', '').strip()
        provides = 'panda3d'
        if install_python_versions:
            depends += ', ' + ' | '.join(('python' + version_info['version'] for version_info in install_python_versions))
            recommends += ', python3'
            recommends += ', python3-tk'
            provides += ', python3-panda3d'
        if not PkgSkip('NVIDIACG'):
            depends += ', nvidia-cg-toolkit'
        txt = txt.replace('DEPENDS', depends.strip(', '))
        txt = txt.replace('RECOMMENDS', recommends.strip(', '))
        txt = txt.replace('PROVIDES', provides.strip(', '))
        WriteFile('targetroot/DEBIAN/control', txt)
        oscmd('rm -rf targetroot/debian')
        oscmd('chmod -R 755 targetroot/DEBIAN')
        oscmd('chmod 644 targetroot/DEBIAN/control targetroot/DEBIAN/md5sums')
        oscmd('chmod 644 targetroot/DEBIAN/conffiles targetroot/DEBIAN/symbols')
        oscmd('fakeroot dpkg-deb -Zxz -b targetroot %s_%s_%s.deb' % (pkg_name, pkg_version, pkg_arch))
    elif rpmbuild_present:
        InstallPanda(destdir='targetroot', prefix='/usr', outputdir=outputdir, libdir=GetRPMLibDir(), python_versions=install_python_versions)
        oscmd('chmod -R 755 targetroot/usr/share/panda3d')
        oscmd("rpm -E '%_target_cpu' > " + outputdir + '/tmp/architecture.txt')
        arch = ReadFile(outputdir + '/tmp/architecture.txt').strip()
        pandasource = os.path.abspath(os.getcwd())
        txt = INSTALLER_SPEC_FILE[1:]
        if not PkgSkip('PVIEW'):
            txt += INSTALLER_SPEC_FILE_PVIEW
        dirs = set()
        for version_info in install_python_versions:
            dirs.add(version_info['platlib'])
            dirs.add(version_info['purelib'])
        for dir in dirs:
            txt += dir + '\n'
        for base in os.listdir(outputdir + '/bin'):
            if not base.startswith('deploy-stub'):
                txt += '/usr/bin/%s\n' % base
        txt = txt.replace('VERSION', rpmversion)
        txt = txt.replace('RPMRELEASE', str(rpmrelease))
        txt = txt.replace('PANDASOURCE', pandasource)
        WriteFile('panda3d.spec', txt)
        oscmd("fakeroot rpmbuild --define '_rpmdir " + pandasource + "' --buildroot '" + os.path.abspath('targetroot') + "' -bb panda3d.spec")
        oscmd('mv ' + arch + '/panda3d-' + rpmversion + '-' + rpmrelease + '.' + arch + '.rpm .')
        oscmd('rm -rf ' + arch, True)
    else:
        exit('To build an installer, either rpmbuild or dpkg-deb must be present on your system!')

def MakeInstallerOSX(version, python_versions=[], installdir=None, **kwargs):
    if False:
        return 10
    outputdir = GetOutputDir()
    if installdir is None:
        installdir = '/Library/Developer/Panda3D'
    dmg_name = 'Panda3D-' + version
    if len(python_versions) == 1 and (not python_versions[0]['version'].startswith('2.')):
        dmg_name += '-py' + python_versions[0]['version']
    dmg_name += '.dmg'
    if os.path.isfile(dmg_name):
        oscmd('rm -f %s' % dmg_name)
    if os.path.exists('dstroot'):
        oscmd('rm -rf dstroot')
    if os.path.exists('Panda3D-rw.dmg'):
        oscmd('rm -f Panda3D-rw.dmg')
    oscmd('mkdir -p                       dstroot/base/%s/lib' % installdir)
    oscmd('mkdir -p                       dstroot/base/%s/etc' % installdir)
    oscmd('cp %s/etc/Config.prc           dstroot/base/%s/etc/Config.prc' % (outputdir, installdir))
    oscmd('cp %s/etc/Confauto.prc         dstroot/base/%s/etc/Confauto.prc' % (outputdir, installdir))
    oscmd('cp -R %s/models                dstroot/base/%s/models' % (outputdir, installdir))
    oscmd('cp -R doc/LICENSE              dstroot/base/%s/LICENSE' % installdir)
    oscmd('cp -R doc/ReleaseNotes         dstroot/base/%s/ReleaseNotes' % installdir)
    if os.path.isdir(outputdir + '/Frameworks') and os.listdir(outputdir + '/Frameworks'):
        oscmd('cp -R %s/Frameworks            dstroot/base/%s/Frameworks' % (outputdir, installdir))
    if os.path.isdir(outputdir + '/plugins'):
        oscmd('cp -R %s/plugins           dstroot/base/%s/plugins' % (outputdir, installdir))
    no_base_libs = ['libp3ffmpeg', 'libp3fmod_audio', 'libfmodex', 'libfmodexL']
    for base in os.listdir(outputdir + '/lib'):
        if not base.endswith('.a') and base.split('.')[0] not in no_base_libs:
            libname = 'dstroot/base/%s/lib/' % installdir + base
            oscmd('cp -R -P ' + outputdir + '/lib/' + base + ' ' + libname)
    oscmd('mkdir -p dstroot/tools/%s/bin' % installdir)
    oscmd('mkdir -p dstroot/tools/etc/paths.d')
    WriteFile('dstroot/tools/etc/paths.d/Panda3D', '/%s/bin\n' % installdir)
    oscmd('mkdir -m 0755 -p dstroot/tools/usr/local/share/man/man1')
    oscmd('install -m 0644 doc/man/*.1 dstroot/tools/usr/local/share/man/man1/')
    for base in os.listdir(outputdir + '/bin'):
        if not base.startswith('deploy-stub'):
            binname = 'dstroot/tools/%s/bin/' % installdir + base
            oscmd('cp -R ' + outputdir + '/bin/' + base + ' ' + binname)
    if python_versions:
        if len(python_versions) == 1:
            oscmd('mkdir -p dstroot/pythoncode/usr/local/bin')
            oscmd('ln -s %s dstroot/pythoncode/usr/local/bin/ppython' % python_versions[0]['executable'])
        oscmd('mkdir -p dstroot/pythoncode/%s/panda3d' % installdir)
        oscmd('cp -R %s/pandac                dstroot/pythoncode/%s/pandac' % (outputdir, installdir))
        oscmd('cp -R %s/direct                dstroot/pythoncode/%s/direct' % (outputdir, installdir))
        oscmd('cp -R %s/*.so                  dstroot/pythoncode/%s/' % (outputdir, installdir), True)
        oscmd('cp -R %s/*.py                  dstroot/pythoncode/%s/' % (outputdir, installdir), True)
        if os.path.isdir(outputdir + '/Pmw'):
            oscmd('cp -R %s/Pmw               dstroot/pythoncode/%s/Pmw' % (outputdir, installdir))
        if os.path.isdir(outputdir + '/panda3d.dist-info'):
            oscmd('cp -R %s/panda3d.dist-info dstroot/pythoncode/%s/panda3d.dist-info' % (outputdir, installdir))
        for base in os.listdir(outputdir + '/panda3d'):
            if base.endswith('.py'):
                libname = 'dstroot/pythoncode/%s/panda3d/' % installdir + base
                oscmd('cp -R ' + outputdir + '/panda3d/' + base + ' ' + libname)
    for version_info in python_versions:
        pyver = version_info['version']
        oscmd('mkdir -p dstroot/pybindings%s/Library/Python/%s/site-packages' % (pyver, pyver))
        oscmd('mkdir -p dstroot/pybindings%s/%s/panda3d' % (pyver, installdir))
        suffix = version_info['ext_suffix']
        for base in os.listdir(outputdir + '/panda3d'):
            if base.endswith(suffix) and '.' not in base[:-len(suffix)]:
                libname = 'dstroot/pybindings%s/%s/panda3d/%s' % (pyver, installdir, base)
                oscmd('cp -R -P ' + outputdir + '/panda3d/' + base + ' ' + libname)
        oscmd('mkdir -p dstroot/pybindings%s/Library/Python/%s/site-packages' % (pyver, pyver))
        WriteFile('dstroot/pybindings%s/Library/Python/%s/site-packages/Panda3D.pth' % (pyver, pyver), installdir)
        if pyver not in ('3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6'):
            dir = 'dstroot/pybindings%s/Library/Frameworks/Python.framework/Versions/%s/lib/python%s/site-packages' % (pyver, pyver, pyver)
            oscmd('mkdir -p %s' % dir)
            WriteFile('%s/Panda3D.pth' % dir, installdir)
        dir = 'dstroot/pybindings%s/usr/local/lib/python%s/site-packages' % (pyver, pyver)
        oscmd('mkdir -p %s' % dir)
        WriteFile('%s/Panda3D.pth' % dir, installdir)
    if not PkgSkip('FFMPEG'):
        oscmd('mkdir -p dstroot/ffmpeg/%s/lib' % installdir)
        oscmd('cp -R %s/lib/libp3ffmpeg.* dstroot/ffmpeg/%s/lib/' % (outputdir, installdir))
    if not PkgSkip('FMODEX'):
        oscmd('mkdir -p dstroot/fmodex/%s/lib' % installdir)
        oscmd('cp -R %s/lib/libp3fmod_audio.* dstroot/fmodex/%s/lib/' % (outputdir, installdir))
        oscmd('cp -R %s/lib/libfmodex* dstroot/fmodex/%s/lib/' % (outputdir, installdir))
    oscmd('mkdir -p dstroot/headers/%s/lib' % installdir)
    oscmd('cp -R %s/include               dstroot/headers/%s/include' % (outputdir, installdir))
    if os.path.isdir('samples'):
        oscmd('mkdir -p dstroot/samples/%s/samples' % installdir)
        oscmd('cp -R samples/* dstroot/samples/%s/samples' % installdir)
    DeleteVCS('dstroot')
    DeleteBuildFiles('dstroot')
    for version_info in python_versions:
        if os.path.isdir('dstroot/pythoncode/%s/Pmw' % installdir):
            oscmd('%s -m compileall -q -f -d %s/Pmw dstroot/pythoncode/%s/Pmw' % (version_info['executable'], installdir, installdir), True)
        oscmd('%s -m compileall -q -f -d %s/direct dstroot/pythoncode/%s/direct' % (version_info['executable'], installdir, installdir))
        oscmd('%s -m compileall -q -f -d %s/pandac dstroot/pythoncode/%s/pandac' % (version_info['executable'], installdir, installdir))
        oscmd('%s -m compileall -q -f -d %s/panda3d dstroot/pythoncode/%s/panda3d' % (version_info['executable'], installdir, installdir))
    oscmd('chmod -R 0775 dstroot/*')
    oscmd('mkdir -p dstroot/Panda3D/Panda3D.mpkg/Contents/Packages/')
    oscmd('mkdir -p dstroot/Panda3D/Panda3D.mpkg/Contents/Resources/en.lproj/')
    pkgs = ['base', 'tools', 'headers']
    script_components = set()

    def write_script(component, phase, contents):
        if False:
            for i in range(10):
                print('nop')
        if installdir == '/Developer/Panda3D':
            return
        script_components.add(component)
        oscmd('mkdir -p dstroot/scripts/%s' % component)
        ln_script = open('dstroot/scripts/%s/%s' % (component, phase), 'w')
        ln_script.write(MACOS_SCRIPT_PREFIX)
        ln_script.write(contents)
        ln_script.write(MACOS_SCRIPT_POSTFIX)
        ln_script.close()
        oscmd('chmod +x dstroot/scripts/%s/%s' % (component, phase))
    write_script('base', 'postinstall', '\n        pkgutil --pkg-info org.panda3d.panda3d.base.pkg\n        if [ $? = 0 ]; then\n            rm -rf /Developer/Panda3D\n        fi\n        mkdir -p /Developer\n        ln -s %s /Developer/Panda3D\n    ' % installdir)
    write_script('tools', 'postinstall', '\n        pkgutil --pkg-info org.panda3d.panda3d.tools.pkg\n        if [ $? = 0 ]; then\n            rm -f /Developer/Tools/Panda3D\n        fi\n        mkdir -p /Developer/Tools\n        ln -s %s/bin /Developer/Tools/Panda3D\n    ' % installdir)
    if os.path.isdir('samples'):
        pkgs.append('samples')
        write_script('samples', 'postinstall', '\n            pkgutil --pkg-info org.panda3d.panda3d.samples.pkg\n            if [ $? = 0 ]; then\n                rm -f /Developer/Examples/Panda3D\n            fi\n            mkdir -p /Developer/Examples\n            ln -s %s/samples /Developer/Examples/Panda3D\n        ' % installdir)
    if python_versions:
        pkgs.append('pythoncode')
    for version_info in python_versions:
        pkgs.append('pybindings' + version_info['version'])
    if not PkgSkip('FFMPEG'):
        pkgs.append('ffmpeg')
    if not PkgSkip('FMODEX'):
        pkgs.append('fmodex')
    for pkg in pkgs:
        identifier = 'org.panda3d.panda3d.%s.pkg' % pkg
        scripts_path = 'dstroot/scripts/%s' % pkg
        plist = open('/tmp/Info_plist', 'w')
        plist.write(Info_plist.format(package_id=identifier, version=version))
        plist.close()
        if not os.path.isdir('dstroot/' + pkg):
            os.makedirs('dstroot/' + pkg)
        if pkg in script_components:
            pkg_scripts = ' --scripts ' + scripts_path
        else:
            pkg_scripts = ''
        if os.path.exists('/usr/bin/pkgbuild'):
            cmd = f'/usr/bin/pkgbuild --identifier {identifier} --version {version} --root dstroot/{pkg}/ dstroot/Panda3D/Panda3D.mpkg/Contents/Packages/{pkg}.pkg {pkg_scripts}'
        else:
            exit('pkgbuild could not be found!')
        oscmd(cmd)
    if os.path.isfile('/tmp/Info_plist'):
        oscmd('rm -f /tmp/Info_plist')
    dist = open('dstroot/Panda3D/Panda3D.mpkg/Contents/distribution.dist', 'w')
    dist.write('<?xml version="1.0" encoding="utf-8"?>\n')
    dist.write('<installer-script minSpecVersion="1.000000" authoringTool="com.apple.PackageMaker" authoringToolVersion="3.0.3" authoringToolBuild="174">\n')
    dist.write('    <title>Panda3D SDK %s</title>\n' % version)
    dist.write('    <allowed-os-versions>\n')
    dist.write('        <os-version min="10.9"/>\n')
    dist.write('    </allowed-os-versions>\n')
    dist.write('    <options customize="always" allow-external-scripts="no" rootVolumeOnly="false" hostArchitectures="x86_64"/>\n')
    dist.write('    <license language="en" mime-type="text/plain">%s</license>\n' % ReadFile('doc/LICENSE'))
    dist.write('    <readme language="en" mime-type="text/plain">')
    dist.write('WARNING: From Panda3D version 1.10.5 onwards, the default installation has been changed from /Developer/Panda3D to /Library/Developer/Panda3D\n')
    dist.write('This installation script will remove any existing installation in /Developer and if possible create a symbolic link towards /Library/Developer/Panda3D\n')
    dist.write('    </readme>')
    dist.write('    <script>\n')
    dist.write('    function isPythonVersionInstalled(version) {\n')
    dist.write('        return system.files.fileExistsAtPath("/usr/bin/python" + version)\n')
    dist.write('            || system.files.fileExistsAtPath("/usr/local/bin/python" + version)\n')
    dist.write('            || system.files.fileExistsAtPath("/opt/local/bin/python" + version)\n')
    dist.write('            || system.files.fileExistsAtPath("/sw/bin/python" + version)\n')
    dist.write('            || system.files.fileExistsAtPath("/System/Library/Frameworks/Python.framework/Versions/" + version + "/bin/python")\n')
    dist.write('            || system.files.fileExistsAtPath("/Library/Frameworks/Python.framework/Versions/" + version + "/bin/python");\n')
    dist.write('    }\n')
    dist.write('    </script>\n')
    dist.write('    <choices-outline>\n')
    dist.write('        <line choice="base"/>\n')
    if python_versions:
        dist.write('        <line choice="pythoncode">\n')
        for version_info in sorted(python_versions, key=lambda info: info['version'], reverse=True):
            dist.write('            <line choice="pybindings%s"/>\n' % version_info['version'])
        dist.write('        </line>\n')
    dist.write('        <line choice="tools"/>\n')
    if os.path.isdir('samples'):
        dist.write('        <line choice="samples"/>\n')
    if not PkgSkip('FFMPEG'):
        dist.write('        <line choice="ffmpeg"/>\n')
    if not PkgSkip('FMODEX'):
        dist.write('        <line choice="fmodex"/>\n')
    dist.write('        <line choice="headers"/>\n')
    dist.write('    </choices-outline>\n')
    dist.write('    <choice id="base" title="Panda3D Base Installation" description="This package contains the Panda3D libraries, configuration files and models/textures that are needed to use Panda3D.&#10;&#10;Location: %s/" start_enabled="false">\n' % installdir)
    dist.write('        <pkg-ref id="org.panda3d.panda3d.base.pkg"/>\n')
    dist.write('    </choice>\n')
    dist.write('    <choice id="tools" title="Tools" tooltip="Useful tools and model converters to help with Panda3D development" description="This package contains the various utilities that ship with Panda3D, including packaging tools, model converters, and many more.&#10;&#10;Location: %s/bin/">\n' % installdir)
    dist.write('        <pkg-ref id="org.panda3d.panda3d.tools.pkg"/>\n')
    dist.write('    </choice>\n')
    if python_versions:
        dist.write('    <choice id="pythoncode" title="Python Support" tooltip="Python bindings for the Panda3D libraries" description="This package contains the \'direct\', \'pandac\' and \'panda3d\' python packages that are needed to do Python development with Panda3D.&#10;&#10;Location: %s/">\n' % installdir)
        dist.write('        <pkg-ref id="org.panda3d.panda3d.pythoncode.pkg"/>\n')
        dist.write('    </choice>\n')
    for version_info in python_versions:
        pyver = version_info['version']
        cond = "isPythonVersionInstalled('%s')" % pyver
        dist.write('    <choice id="pybindings%s" start_selected="%s" title="Python %s Bindings" tooltip="Python bindings for the Panda3D libraries" description="Support for Python %s.">\n' % (pyver, cond, pyver, pyver))
        dist.write('        <pkg-ref id="org.panda3d.panda3d.pybindings%s.pkg"/>\n' % pyver)
        dist.write('    </choice>\n')
    if not PkgSkip('FFMPEG'):
        dist.write('    <choice id="ffmpeg" title="FFMpeg Plug-In" tooltip="FFMpeg video and audio decoding plug-in" description="This package contains the FFMpeg plug-in, which is used for decoding video and audio files with OpenAL.')
        if PkgSkip('VORBIS') and PkgSkip('OPUS'):
            dist.write('  It is not required for loading .wav files, which Panda3D can read out of the box.">\n')
        elif PkgSkip('VORBIS'):
            dist.write('  It is not required for loading .wav or .opus files, which Panda3D can read out of the box.">\n')
        elif PkgSkip('OPUS'):
            dist.write('  It is not required for loading .wav or .ogg files, which Panda3D can read out of the box.">\n')
        else:
            dist.write('  It is not required for loading .wav, .ogg or .opus files, which Panda3D can read out of the box.">\n')
        dist.write('        <pkg-ref id="org.panda3d.panda3d.ffmpeg.pkg"/>\n')
        dist.write('    </choice>\n')
    if not PkgSkip('FMODEX'):
        dist.write('    <choice id="fmodex" title="FMOD Ex Plug-In" tooltip="FMOD Ex audio output plug-in" description="This package contains the FMOD Ex audio plug-in, which is a commercial library for playing sounds.  It is an optional component as Panda3D can use the open-source alternative OpenAL instead.">\n')
        dist.write('        <pkg-ref id="org.panda3d.panda3d.fmodex.pkg"/>\n')
        dist.write('    </choice>\n')
    if os.path.isdir('samples'):
        dist.write('    <choice id="samples" title="Sample Programs" tooltip="Python sample programs that use Panda3D" description="This package contains the Python sample programs that can help you with learning how to use Panda3D.&#10;&#10;Location: %s/samples">\n' % installdir)
        dist.write('        <pkg-ref id="org.panda3d.panda3d.samples.pkg"/>\n')
        dist.write('    </choice>\n')
    dist.write('    <choice id="headers" title="C++ Header Files" tooltip="Header files for C++ development with Panda3D" description="This package contains the C++ header files that are needed in order to do C++ development with Panda3D. You don\'t need this if you want to develop in Python.&#10;&#10;Location: %s/include/" start_selected="false">\n' % installdir)
    dist.write('        <pkg-ref id="org.panda3d.panda3d.headers.pkg"/>\n')
    dist.write('    </choice>\n')
    for pkg in pkgs:
        size = GetDirectorySize('dstroot/' + pkg) // 1024
        dist.write('    <pkg-ref id="org.panda3d.panda3d.%s.pkg" installKBytes="%d" version="1" auth="Root">file:./Contents/Packages/%s.pkg</pkg-ref>\n' % (pkg, size, pkg))
    dist.write('</installer-script>\n')
    dist.close()
    oscmd('hdiutil create Panda3D-rw.dmg -fs HFS+ -volname "Panda3D SDK %s" -srcfolder dstroot/Panda3D' % version)
    oscmd('hdiutil convert Panda3D-rw.dmg -format UDBZ -o %s' % dmg_name)
    oscmd('rm -f Panda3D-rw.dmg')

def MakeInstallerFreeBSD(version, python_versions=[], **kwargs):
    if False:
        i = 10
        return i + 15
    outputdir = GetOutputDir()
    oscmd('rm -rf targetroot +DESC pkg-plist +MANIFEST')
    oscmd('mkdir targetroot')
    InstallPanda(destdir='targetroot', prefix='/usr/local', outputdir=outputdir, python_versions=python_versions)
    if not os.path.exists('/usr/sbin/pkg'):
        exit('Cannot create an installer without pkg')
    plist_txt = ''
    for (root, dirs, files) in os.walk('targetroot/usr/local/', True):
        for f in files:
            plist_txt += os.path.join(root, f)[21:] + '\n'
    plist_txt += '@postexec /sbin/ldconfig -m /usr/local/lib/panda3d\n'
    plist_txt += '@postunexec /sbin/ldconfig -R\n'
    for remdir in ('lib/panda3d', 'share/panda3d', 'include/panda3d'):
        for (root, dirs, files) in os.walk('targetroot/usr/local/' + remdir, False):
            for d in dirs:
                plist_txt += '@dir %s\n' % os.path.join(root, d)[21:]
        plist_txt += '@dir %s\n' % remdir
    oscmd('echo "`pkg config abi | tr \'[:upper:]\' \'[:lower:]\' | cut -d: -f1,2`:*" > ' + outputdir + '/tmp/architecture.txt')
    pkg_arch = ReadFile(outputdir + '/tmp/architecture.txt').strip()
    dependencies = ''
    if not PkgSkip('PYTHON'):
        oscmd('rm -f %s/tmp/python_dep' % outputdir)
        if 'PYTHONVERSION' in SDK:
            pyver_nodot = SDK['PYTHONVERSION'][6:].rstrip('dmu').replace('.', '')
        else:
            pyver_nodot = '%d%d' % sys.version_info[:2]
        oscmd('pkg query "\n\t%%n : {\n\t\torigin : %%o,\n\t\tversion : %%v\n\t},\n" python%s > %s/tmp/python_dep' % (pyver_nodot, outputdir), True)
        if os.path.isfile(outputdir + '/tmp/python_dep'):
            python_pkg = ReadFile(outputdir + '/tmp/python_dep')
            if python_pkg:
                dependencies += python_pkg
    manifest_txt = INSTALLER_PKG_MANIFEST_FILE[1:].replace('NAME', 'panda3d')
    manifest_txt = manifest_txt.replace('VERSION', version)
    manifest_txt = manifest_txt.replace('ARCH', pkg_arch)
    manifest_txt = manifest_txt.replace('ORIGIN', 'devel/panda3d')
    manifest_txt = manifest_txt.replace('DEPENDS', dependencies)
    manifest_txt = manifest_txt.replace('INSTSIZE', str(GetDirectorySize('targetroot') // 1024 // 1024))
    WriteFile('pkg-plist', plist_txt)
    WriteFile('+DESC', INSTALLER_PKG_DESCR_FILE[1:])
    WriteFile('+MANIFEST', manifest_txt)
    oscmd('pkg create -p pkg-plist -r %s  -m . -o . %s' % (os.path.abspath('targetroot'), '--verbose' if GetVerbose() else '--quiet'))

def MakeInstallerAndroid(version, **kwargs):
    if False:
        i = 10
        return i + 15
    outputdir = GetOutputDir()
    oscmd('rm -rf apkroot')
    oscmd('mkdir apkroot')
    apk_unaligned = os.path.join(outputdir, 'tmp', 'panda3d-unaligned.apk')
    apk_unsigned = os.path.join(outputdir, 'tmp', 'panda3d-unsigned.apk')
    if os.path.exists(apk_unaligned):
        os.unlink(apk_unaligned)
    if os.path.exists(apk_unsigned):
        os.unlink(apk_unsigned)
    oscmd('cp %s apkroot/classes.dex' % os.path.join(outputdir, 'classes.dex'))
    source_dir = os.path.join(outputdir, 'lib')
    target_dir = os.path.join('apkroot', 'lib', SDK['ANDROID_ABI'])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, mode=493)
    libpath = [source_dir]
    for dir in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        dir = os.path.expandvars(dir)
        dir = os.path.expanduser(dir)
        if os.path.isdir(dir):
            dir = os.path.realpath(dir)
            if not dir.startswith('/system') and (not dir.startswith('/vendor')):
                libpath.append(dir)

    def copy_library(source, base):
        if False:
            print('Hello World!')
        target = os.path.join(target_dir, base)
        if not target.endswith('.so'):
            target = target.rpartition('.so.')[0] + '.so'
        if os.path.isfile(target):
            return
        shutil.copy(source, target)
        handle = subprocess.Popen(['readelf', '--dynamic', target], stdout=subprocess.PIPE)
        for line in handle.communicate()[0].splitlines():
            line = line.decode('utf-8', 'replace').strip()
            if not line or '(NEEDED)' not in line or '[' not in line or (']' not in line):
                continue
            idx = line.index('[')
            dep = line[idx + 1:line.index(']', idx)]
            if '.so.' in dep:
                orig_dep = dep
                dep = dep.rpartition('.so.')[0] + '.so'
                oscmd('patchelf --replace-needed %s %s %s' % (orig_dep, dep, target), True)
            for dir in libpath:
                fulldep = os.path.join(dir, dep)
                if os.path.isfile(fulldep):
                    copy_library(os.path.realpath(fulldep), dep)
                    break
    for base in os.listdir(source_dir):
        if not base.startswith('lib'):
            continue
        if not base.endswith('.so') and '.so.' not in base:
            continue
        source = os.path.join(source_dir, base)
        if os.path.islink(source):
            continue
        copy_library(source, base)
    if not PkgSkip('PYTHON'):
        suffix = GetExtensionSuffix()
        source_dir = os.path.join(outputdir, 'panda3d')
        for base in os.listdir(source_dir):
            if not base.endswith(suffix):
                continue
            modname = base[:-len(suffix)]
            if '.' not in modname:
                source = os.path.join(source_dir, base)
                copy_library(source, 'libpy.panda3d.{}.so'.format(modname))
        if CrossCompiling():
            source_dir = os.path.join(GetThirdpartyDir(), 'python', 'lib', SDK['PYTHONVERSION'], 'lib-dynload')
        else:
            import _ctypes
            source_dir = os.path.dirname(_ctypes.__file__)
        for base in os.listdir(source_dir):
            if not base.endswith('.so'):
                continue
            modname = base.partition('.')[0]
            source = os.path.join(source_dir, base)
            copy_library(source, 'libpy.{}.so'.format(modname))

    def copy_python_tree(source_root, target_root):
        if False:
            return 10
        for (source_dir, dirs, files) in os.walk(source_root):
            if 'site-packages' in dirs:
                dirs.remove('site-packages')
            if not any((base.endswith('.py') for base in files)):
                continue
            target_dir = os.path.join(target_root, os.path.relpath(source_dir, source_root))
            target_dir = os.path.normpath(target_dir)
            os.makedirs(target_dir, 493)
            for base in files:
                if base.endswith('.py'):
                    target = os.path.join(target_dir, base)
                    shutil.copy(os.path.join(source_dir, base), target)
    from locations import get_python_lib
    stdlib_source = get_python_lib(False, True)
    stdlib_target = os.path.join('apkroot', 'lib', 'python{0}.{1}'.format(*sys.version_info))
    copy_python_tree(stdlib_source, stdlib_target)
    shutil.copy('panda/src/android/site.py', os.path.join(stdlib_target, 'site.py'))
    for tree in ('panda3d', 'direct', 'pandac'):
        copy_python_tree(os.path.join(outputdir, tree), os.path.join(stdlib_target, 'site-packages', tree))
    oscmd('mkdir apkroot/assets')
    oscmd('cp -R %s apkroot/assets/models' % os.path.join(outputdir, 'models'))
    oscmd('cp -R %s apkroot/assets/etc' % os.path.join(outputdir, 'etc'))
    oscmd('mkdir apkroot/res')
    oscmd('cp panda/src/android/pview_manifest.xml apkroot/AndroidManifest.xml')
    aapt_cmd = 'aapt package'
    aapt_cmd += ' -F %s' % apk_unaligned
    aapt_cmd += ' -M apkroot/AndroidManifest.xml'
    aapt_cmd += ' -A apkroot/assets -S apkroot/res'
    aapt_cmd += ' -I %s' % SDK['ANDROID_JAR']
    oscmd(aapt_cmd)
    oscmd('aapt add %s classes.dex' % os.path.join('..', apk_unaligned), cwd='apkroot')
    for (path, dirs, files) in os.walk('apkroot/lib'):
        if files:
            rel = os.path.relpath(path, 'apkroot')
            rel_files = [os.path.join(rel, file).replace('\\', '/') for file in files]
            oscmd('aapt add %s %s' % (os.path.join('..', apk_unaligned), ' '.join(rel_files)), cwd='apkroot')
    oscmd('zipalign -v -p 4 %s %s' % (apk_unaligned, apk_unsigned))
    if GetHost() == 'android':
        oscmd('apksigner debug.ks %s panda3d.apk' % apk_unsigned)
    else:
        if not os.path.isfile('debug.ks'):
            oscmd('keytool -genkey -noprompt -dname CN=Panda3D,O=Panda3D,C=US -keystore debug.ks -storepass android -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 -validity 1000')
        oscmd('apksigner sign --ks debug.ks --ks-pass pass:android --min-sdk-version %s --out panda3d.apk %s' % (SDK['ANDROID_API'], apk_unsigned))
    oscmd('rm -rf apkroot')
    os.unlink(apk_unaligned)
    os.unlink(apk_unsigned)

def MakeInstaller(version, **kwargs):
    if False:
        while True:
            i = 10
    target = GetTarget()
    if target == 'windows':
        dir = kwargs.pop('installdir', None)
        if dir is None:
            dir = 'C:\\Panda3D-' + version
            if GetTargetArch() == 'x64':
                dir += '-x64'
        fn = 'Panda3D-'
        title = 'Panda3D SDK ' + version
        fn += version
        python_versions = kwargs.get('python_versions', [])
        if len(python_versions) == 1:
            fn += '-py' + python_versions[0]['version']
        if GetOptimize() <= 2:
            fn += '-dbg'
        if GetTargetArch() == 'x64':
            fn += '-x64'
        compressor = kwargs.get('compressor')
        MakeInstallerNSIS(version, fn + '.exe', title, dir, **kwargs)
        MakeDebugSymbolArchive(fn + '-pdb', compressor)
    elif target == 'linux':
        MakeInstallerLinux(version, **kwargs)
    elif target == 'darwin':
        MakeInstallerOSX(version, **kwargs)
    elif target == 'freebsd':
        MakeInstallerFreeBSD(version, **kwargs)
    elif target == 'android':
        MakeInstallerAndroid(version, **kwargs)
    else:
        exit('Do not know how to make an installer for this platform')
if __name__ == '__main__':
    version = GetMetadataValue('version')
    parser = OptionParser()
    parser.add_option('', '--version', dest='version', help='Panda3D version number (default: %s)' % version, default=version)
    parser.add_option('', '--debversion', dest='debversion', help='Version number for .deb file', default=None)
    parser.add_option('', '--rpmversion', dest='rpmversion', help='Version number for .rpm file', default=None)
    parser.add_option('', '--rpmrelease', dest='rpmrelease', help='Release number for .rpm file', default='1')
    parser.add_option('', '--outputdir', dest='outputdir', help="Makepanda's output directory (default: built)", default='built')
    parser.add_option('', '--verbose', dest='verbose', help='Enable verbose output', action='store_true', default=False)
    parser.add_option('', '--lzma', dest='compressor', help='Use LZMA compression', action='store_const', const='lzma', default='zlib')
    parser.add_option('', '--installdir', dest='installdir', help='Where on the system the installer should put the SDK (Windows, macOS)')
    (options, args) = parser.parse_args()
    SetVerbose(options.verbose)
    SetOutputDir(options.outputdir)
    opt = ReadFile(os.path.join(options.outputdir, 'tmp', 'optimize.dat'))
    SetOptimize(int(opt.strip()))
    pkg_list = ('PYTHON', 'NVIDIACG', 'FFMPEG', 'OPENAL', 'FMODEX', 'PVIEW', 'NVIDIACG', 'VORBIS', 'OPUS')
    PkgListSet(pkg_list)
    for pkg in pkg_list:
        dat_path = 'dtool_have_%s.dat' % pkg.lower()
        content = ReadFile(os.path.join(options.outputdir, 'tmp', dat_path))
        if int(content.strip()):
            PkgEnable(pkg)
        else:
            PkgDisable(pkg)
    match = re.match('^\\d+\\.\\d+(\\.\\d+)+', options.version)
    if not match:
        exit('version requires three digits')
    MakeInstaller(version=match.group(), outputdir=options.outputdir, optimize=GetOptimize(), compressor=options.compressor, debversion=options.debversion, rpmversion=options.rpmversion, rpmrelease=options.rpmrelease, python_versions=ReadPythonVersionInfoFile(), installdir=options.installdir)