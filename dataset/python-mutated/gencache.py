"""Manages the cache of generated Python code.

Description
  This file manages the cache of generated Python code.  When run from the
  command line, it also provides a number of options for managing that cache.

Implementation
  Each typelib is generated into a filename of format "{guid}x{lcid}x{major}x{minor}.py"

  An external persistant dictionary maps from all known IIDs in all known type libraries
  to the type library itself.

  Thus, whenever Python code knows the IID of an object, it can find the IID, LCID and version of
  the type library which supports it.  Given this information, it can find the Python module
  with the support.

  If necessary, this support can be generated on the fly.

Hacks, to do, etc
  Currently just uses a pickled dictionary, but should used some sort of indexed file.
  Maybe an OLE2 compound file, or a bsddb file?
"""
import glob
import os
import sys
from importlib import reload
import pythoncom
import pywintypes
import win32com
import win32com.client
from . import CLSIDToClass
bForDemandDefault = 0
clsidToTypelib = {}
versionRedirectMap = {}
is_readonly = is_zip = hasattr(win32com, '__loader__') and hasattr(win32com.__loader__, 'archive')
demandGeneratedTypeLibraries = {}
import pickle as pickle

def __init__():
    if False:
        for i in range(10):
            print('nop')
    try:
        _LoadDicts()
    except OSError:
        Rebuild()
pickleVersion = 1

def _SaveDicts():
    if False:
        for i in range(10):
            print('nop')
    if is_readonly:
        raise RuntimeError("Trying to write to a readonly gencache ('%s')!" % win32com.__gen_path__)
    f = open(os.path.join(GetGeneratePath(), 'dicts.dat'), 'wb')
    try:
        p = pickle.Pickler(f)
        p.dump(pickleVersion)
        p.dump(clsidToTypelib)
    finally:
        f.close()

def _LoadDicts():
    if False:
        for i in range(10):
            print('nop')
    if is_zip:
        import io as io
        loader = win32com.__loader__
        arc_path = loader.archive
        dicts_path = os.path.join(win32com.__gen_path__, 'dicts.dat')
        if dicts_path.startswith(arc_path):
            dicts_path = dicts_path[len(arc_path) + 1:]
        else:
            return
        try:
            data = loader.get_data(dicts_path)
        except AttributeError:
            return
        except OSError:
            return
        f = io.BytesIO(data)
    else:
        f = open(os.path.join(win32com.__gen_path__, 'dicts.dat'), 'rb')
    try:
        p = pickle.Unpickler(f)
        version = p.load()
        global clsidToTypelib
        clsidToTypelib = p.load()
        versionRedirectMap.clear()
    finally:
        f.close()

def GetGeneratedFileName(clsid, lcid, major, minor):
    if False:
        return 10
    'Given the clsid, lcid, major and  minor for a type lib, return\n    the file name (no extension) providing this support.\n    '
    return str(clsid).upper()[1:-1] + f'x{lcid}x{major}x{minor}'

def SplitGeneratedFileName(fname):
    if False:
        for i in range(10):
            print('nop')
    'Reverse of GetGeneratedFileName()'
    return tuple(fname.split('x', 4))

def GetGeneratePath():
    if False:
        print('Hello World!')
    'Returns the name of the path to generate to.\n    Checks the directory is OK.\n    '
    assert not is_readonly, 'Why do you want the genpath for a readonly store?'
    try:
        os.makedirs(win32com.__gen_path__)
    except OSError:
        pass
    try:
        fname = os.path.join(win32com.__gen_path__, '__init__.py')
        os.stat(fname)
    except OSError:
        f = open(fname, 'w')
        f.write('# Generated file - this directory may be deleted to reset the COM cache...\n')
        f.write('import win32com\n')
        f.write('if __path__[:-1] != win32com.__gen_path__: __path__.append(win32com.__gen_path__)\n')
        f.close()
    return win32com.__gen_path__

def GetClassForProgID(progid):
    if False:
        print('Hello World!')
    'Get a Python class for a Program ID\n\n    Given a Program ID, return a Python class which wraps the COM object\n\n    Returns the Python class, or None if no module is available.\n\n    Params\n    progid -- A COM ProgramID or IID (eg, "Word.Application")\n    '
    clsid = pywintypes.IID(progid)
    return GetClassForCLSID(clsid)

def GetClassForCLSID(clsid):
    if False:
        i = 10
        return i + 15
    'Get a Python class for a CLSID\n\n    Given a CLSID, return a Python class which wraps the COM object\n\n    Returns the Python class, or None if no module is available.\n\n    Params\n    clsid -- A COM CLSID (or string repr of one)\n    '
    clsid = str(clsid)
    if CLSIDToClass.HasClass(clsid):
        return CLSIDToClass.GetClass(clsid)
    mod = GetModuleForCLSID(clsid)
    if mod is None:
        return None
    try:
        return CLSIDToClass.GetClass(clsid)
    except KeyError:
        return None

def GetModuleForProgID(progid):
    if False:
        i = 10
        return i + 15
    'Get a Python module for a Program ID\n\n    Given a Program ID, return a Python module which contains the\n    class which wraps the COM object.\n\n    Returns the Python module, or None if no module is available.\n\n    Params\n    progid -- A COM ProgramID or IID (eg, "Word.Application")\n    '
    try:
        iid = pywintypes.IID(progid)
    except pywintypes.com_error:
        return None
    return GetModuleForCLSID(iid)

def GetModuleForCLSID(clsid):
    if False:
        while True:
            i = 10
    'Get a Python module for a CLSID\n\n    Given a CLSID, return a Python module which contains the\n    class which wraps the COM object.\n\n    Returns the Python module, or None if no module is available.\n\n    Params\n    progid -- A COM CLSID (ie, not the description)\n    '
    clsid_str = str(clsid)
    try:
        (typelibCLSID, lcid, major, minor) = clsidToTypelib[clsid_str]
    except KeyError:
        return None
    try:
        mod = GetModuleForTypelib(typelibCLSID, lcid, major, minor)
    except ImportError:
        mod = None
    if mod is not None:
        sub_mod = mod.CLSIDToPackageMap.get(clsid_str)
        if sub_mod is None:
            sub_mod = mod.VTablesToPackageMap.get(clsid_str)
        if sub_mod is not None:
            sub_mod_name = mod.__name__ + '.' + sub_mod
            try:
                __import__(sub_mod_name)
            except ImportError:
                info = (typelibCLSID, lcid, major, minor)
                if info in demandGeneratedTypeLibraries:
                    info = demandGeneratedTypeLibraries[info]
                from . import makepy
                makepy.GenerateChildFromTypeLibSpec(sub_mod, info)
            mod = sys.modules[sub_mod_name]
    return mod

def GetModuleForTypelib(typelibCLSID, lcid, major, minor):
    if False:
        i = 10
        return i + 15
    'Get a Python module for a type library ID\n\n    Given the CLSID of a typelibrary, return an imported Python module,\n    else None\n\n    Params\n    typelibCLSID -- IID of the type library.\n    major -- Integer major version.\n    minor -- Integer minor version\n    lcid -- Integer LCID for the library.\n    '
    modName = GetGeneratedFileName(typelibCLSID, lcid, major, minor)
    mod = _GetModule(modName)
    if '_in_gencache_' not in mod.__dict__:
        AddModuleToCache(typelibCLSID, lcid, major, minor)
        assert '_in_gencache_' in mod.__dict__
    return mod

def MakeModuleForTypelib(typelibCLSID, lcid, major, minor, progressInstance=None, bForDemand=bForDemandDefault, bBuildHidden=1):
    if False:
        return 10
    'Generate support for a type library.\n\n    Given the IID, LCID and version information for a type library, generate\n    and import the necessary support files.\n\n    Returns the Python module.  No exceptions are caught.\n\n    Params\n    typelibCLSID -- IID of the type library.\n    major -- Integer major version.\n    minor -- Integer minor version.\n    lcid -- Integer LCID for the library.\n    progressInstance -- Instance to use as progress indicator, or None to\n                        use the GUI progress bar.\n    '
    from . import makepy
    makepy.GenerateFromTypeLibSpec((typelibCLSID, lcid, major, minor), progressInstance=progressInstance, bForDemand=bForDemand, bBuildHidden=bBuildHidden)
    return GetModuleForTypelib(typelibCLSID, lcid, major, minor)

def MakeModuleForTypelibInterface(typelib_ob, progressInstance=None, bForDemand=bForDemandDefault, bBuildHidden=1):
    if False:
        for i in range(10):
            print('nop')
    'Generate support for a type library.\n\n    Given a PyITypeLib interface generate and import the necessary support files.  This is useful\n    for getting makepy support for a typelibrary that is not registered - the caller can locate\n    and load the type library itself, rather than relying on COM to find it.\n\n    Returns the Python module.\n\n    Params\n    typelib_ob -- The type library itself\n    progressInstance -- Instance to use as progress indicator, or None to\n                        use the GUI progress bar.\n    '
    from . import makepy
    try:
        makepy.GenerateFromTypeLibSpec(typelib_ob, progressInstance=progressInstance, bForDemand=bForDemandDefault, bBuildHidden=bBuildHidden)
    except pywintypes.com_error:
        return None
    tla = typelib_ob.GetLibAttr()
    guid = tla[0]
    lcid = tla[1]
    major = tla[3]
    minor = tla[4]
    return GetModuleForTypelib(guid, lcid, major, minor)

def EnsureModuleForTypelibInterface(typelib_ob, progressInstance=None, bForDemand=bForDemandDefault, bBuildHidden=1):
    if False:
        return 10
    'Check we have support for a type library, generating if not.\n\n    Given a PyITypeLib interface generate and import the necessary\n    support files if necessary. This is useful for getting makepy support\n    for a typelibrary that is not registered - the caller can locate and\n    load the type library itself, rather than relying on COM to find it.\n\n    Returns the Python module.\n\n    Params\n    typelib_ob -- The type library itself\n    progressInstance -- Instance to use as progress indicator, or None to\n                        use the GUI progress bar.\n    '
    tla = typelib_ob.GetLibAttr()
    guid = tla[0]
    lcid = tla[1]
    major = tla[3]
    minor = tla[4]
    if bForDemand:
        demandGeneratedTypeLibraries[str(guid), lcid, major, minor] = typelib_ob
    try:
        return GetModuleForTypelib(guid, lcid, major, minor)
    except ImportError:
        pass
    return MakeModuleForTypelibInterface(typelib_ob, progressInstance, bForDemand, bBuildHidden)

def ForgetAboutTypelibInterface(typelib_ob):
    if False:
        print('Hello World!')
    'Drop any references to a typelib previously added with EnsureModuleForTypelibInterface and forDemand'
    tla = typelib_ob.GetLibAttr()
    guid = tla[0]
    lcid = tla[1]
    major = tla[3]
    minor = tla[4]
    info = (str(guid), lcid, major, minor)
    try:
        del demandGeneratedTypeLibraries[info]
    except KeyError:
        print('ForgetAboutTypelibInterface:: Warning - type library with info {} is not being remembered!'.format(info))
    for (key, val) in list(versionRedirectMap.items()):
        if val == info:
            del versionRedirectMap[key]

def EnsureModule(typelibCLSID, lcid, major, minor, progressInstance=None, bValidateFile=not is_readonly, bForDemand=bForDemandDefault, bBuildHidden=1):
    if False:
        while True:
            i = 10
    'Ensure Python support is loaded for a type library, generating if necessary.\n\n    Given the IID, LCID and version information for a type library, check and if\n    necessary (re)generate, then import the necessary support files. If we regenerate the file, there\n    is no way to totally snuff out all instances of the old module in Python, and thus we will regenerate the file more than necessary,\n    unless makepy/genpy is modified accordingly.\n\n\n    Returns the Python module.  No exceptions are caught during the generate process.\n\n    Params\n    typelibCLSID -- IID of the type library.\n    major -- Integer major version.\n    minor -- Integer minor version\n    lcid -- Integer LCID for the library.\n    progressInstance -- Instance to use as progress indicator, or None to\n                        use the GUI progress bar.\n    bValidateFile -- Whether or not to perform cache validation or not\n    bForDemand -- Should a complete generation happen now, or on demand?\n    bBuildHidden -- Should hidden members/attributes etc be generated?\n    '
    bReloadNeeded = 0
    try:
        try:
            module = GetModuleForTypelib(typelibCLSID, lcid, major, minor)
        except ImportError:
            module = None
            try:
                tlbAttr = pythoncom.LoadRegTypeLib(typelibCLSID, major, minor, lcid).GetLibAttr()
                if tlbAttr[1] != lcid or tlbAttr[4] != minor:
                    try:
                        module = GetModuleForTypelib(typelibCLSID, tlbAttr[1], tlbAttr[3], tlbAttr[4])
                    except ImportError:
                        minor = tlbAttr[4]
            except pythoncom.com_error:
                pass
        if module is not None and bValidateFile:
            assert not is_readonly, "Can't validate in a read-only gencache"
            try:
                typLibPath = pythoncom.QueryPathOfRegTypeLib(typelibCLSID, major, minor, lcid)
                if typLibPath[-1] == '\x00':
                    typLibPath = typLibPath[:-1]
                suf = getattr(os.path, 'supports_unicode_filenames', 0)
                if not suf:
                    try:
                        typLibPath = typLibPath.encode(sys.getfilesystemencoding())
                    except AttributeError:
                        typLibPath = str(typLibPath)
                tlbAttributes = pythoncom.LoadRegTypeLib(typelibCLSID, major, minor, lcid).GetLibAttr()
            except pythoncom.com_error:
                bValidateFile = 0
        if module is not None and bValidateFile:
            assert not is_readonly, "Can't validate in a read-only gencache"
            filePathPrefix = '{}\\{}'.format(GetGeneratePath(), GetGeneratedFileName(typelibCLSID, lcid, major, minor))
            filePath = filePathPrefix + '.py'
            filePathPyc = filePathPrefix + '.py'
            if __debug__:
                filePathPyc = filePathPyc + 'c'
            else:
                filePathPyc = filePathPyc + 'o'
            from . import genpy
            if module.MinorVersion != tlbAttributes[4] or genpy.makepy_version != module.makepy_version:
                try:
                    os.unlink(filePath)
                except OSError:
                    pass
                try:
                    os.unlink(filePathPyc)
                except OSError:
                    pass
                if os.path.isdir(filePathPrefix):
                    import shutil
                    shutil.rmtree(filePathPrefix)
                minor = tlbAttributes[4]
                module = None
                bReloadNeeded = 1
            else:
                minor = module.MinorVersion
                filePathPrefix = '{}\\{}'.format(GetGeneratePath(), GetGeneratedFileName(typelibCLSID, lcid, major, minor))
                filePath = filePathPrefix + '.py'
                filePathPyc = filePathPrefix + '.pyc'
                fModTimeSet = 0
                try:
                    pyModTime = os.stat(filePath)[8]
                    fModTimeSet = 1
                except OSError as e:
                    try:
                        pyModTime = os.stat(filePathPyc)[8]
                        fModTimeSet = 1
                    except OSError as e:
                        pass
                typLibModTime = os.stat(typLibPath)[8]
                if fModTimeSet and typLibModTime > pyModTime:
                    bReloadNeeded = 1
                    module = None
    except (ImportError, OSError):
        module = None
    if module is None:
        if is_readonly:
            key = (str(typelibCLSID), lcid, major, minor)
            try:
                return versionRedirectMap[key]
            except KeyError:
                pass
            items = []
            for desc in GetGeneratedInfos():
                if key[0] == desc[0] and key[1] == desc[1] and (key[2] == desc[2]):
                    items.append(desc)
            if items:
                items.sort()
                new_minor = items[-1][3]
                ret = GetModuleForTypelib(typelibCLSID, lcid, major, new_minor)
            else:
                ret = None
            versionRedirectMap[key] = ret
            return ret
        module = MakeModuleForTypelib(typelibCLSID, lcid, major, minor, progressInstance, bForDemand=bForDemand, bBuildHidden=bBuildHidden)
        if bReloadNeeded:
            module = reload(module)
            AddModuleToCache(typelibCLSID, lcid, major, minor)
    return module

def EnsureDispatch(prog_id, bForDemand=1):
    if False:
        i = 10
        return i + 15
    'Given a COM prog_id, return an object that is using makepy support, building if necessary'
    disp = win32com.client.Dispatch(prog_id)
    if not disp.__dict__.get('CLSID'):
        try:
            ti = disp._oleobj_.GetTypeInfo()
            disp_clsid = ti.GetTypeAttr()[0]
            (tlb, index) = ti.GetContainingTypeLib()
            tla = tlb.GetLibAttr()
            mod = EnsureModule(tla[0], tla[1], tla[3], tla[4], bForDemand=bForDemand)
            GetModuleForCLSID(disp_clsid)
            from . import CLSIDToClass
            disp_class = CLSIDToClass.GetClass(str(disp_clsid))
            disp = disp_class(disp._oleobj_)
        except pythoncom.com_error:
            raise TypeError('This COM object can not automate the makepy process - please run makepy manually for this object')
    return disp

def AddModuleToCache(typelibclsid, lcid, major, minor, verbose=1, bFlushNow=not is_readonly):
    if False:
        print('Hello World!')
    'Add a newly generated file to the cache dictionary.'
    fname = GetGeneratedFileName(typelibclsid, lcid, major, minor)
    mod = _GetModule(fname)
    mod._in_gencache_ = 1
    info = (str(typelibclsid), lcid, major, minor)
    dict_modified = False

    def SetTypelibForAllClsids(dict):
        if False:
            while True:
                i = 10
        nonlocal dict_modified
        for (clsid, cls) in dict.items():
            if clsidToTypelib.get(clsid) != info:
                clsidToTypelib[clsid] = info
                dict_modified = True
    SetTypelibForAllClsids(mod.CLSIDToClassMap)
    SetTypelibForAllClsids(mod.CLSIDToPackageMap)
    SetTypelibForAllClsids(mod.VTablesToClassMap)
    SetTypelibForAllClsids(mod.VTablesToPackageMap)
    if info in versionRedirectMap:
        del versionRedirectMap[info]
    if bFlushNow and dict_modified:
        _SaveDicts()

def GetGeneratedInfos():
    if False:
        i = 10
        return i + 15
    zip_pos = win32com.__gen_path__.find('.zip\\')
    if zip_pos >= 0:
        import zipfile
        zip_file = win32com.__gen_path__[:zip_pos + 4]
        zip_path = win32com.__gen_path__[zip_pos + 5:].replace('\\', '/')
        zf = zipfile.ZipFile(zip_file)
        infos = {}
        for n in zf.namelist():
            if not n.startswith(zip_path):
                continue
            base = n[len(zip_path) + 1:].split('/')[0]
            try:
                (iid, lcid, major, minor) = base.split('x')
                lcid = int(lcid)
                major = int(major)
                minor = int(minor)
                iid = pywintypes.IID('{' + iid + '}')
            except ValueError:
                continue
            except pywintypes.com_error:
                continue
            infos[iid, lcid, major, minor] = 1
        zf.close()
        return list(infos.keys())
    else:
        files = glob.glob(win32com.__gen_path__ + '\\*')
        ret = []
        for file in files:
            if not os.path.isdir(file) and (not os.path.splitext(file)[1] == '.py'):
                continue
            name = os.path.splitext(os.path.split(file)[1])[0]
            try:
                (iid, lcid, major, minor) = name.split('x')
                iid = pywintypes.IID('{' + iid + '}')
                lcid = int(lcid)
                major = int(major)
                minor = int(minor)
            except ValueError:
                continue
            except pywintypes.com_error:
                continue
            ret.append((iid, lcid, major, minor))
        return ret

def _GetModule(fname):
    if False:
        return 10
    'Given the name of a module in the gen_py directory, import and return it.'
    mod_name = 'win32com.gen_py.%s' % fname
    mod = __import__(mod_name)
    return sys.modules[mod_name]

def Rebuild(verbose=1):
    if False:
        i = 10
        return i + 15
    'Rebuild the cache indexes from the file system.'
    clsidToTypelib.clear()
    infos = GetGeneratedInfos()
    if verbose and len(infos):
        print('Rebuilding cache of generated files for COM support...')
    for info in infos:
        (iid, lcid, major, minor) = info
        if verbose:
            print('Checking', GetGeneratedFileName(*info))
        try:
            AddModuleToCache(iid, lcid, major, minor, verbose, 0)
        except:
            print('Could not add module {} - {}: {}'.format(info, sys.exc_info()[0], sys.exc_info()[1]))
    if verbose and len(infos):
        print('Done.')
    _SaveDicts()

def _Dump():
    if False:
        for i in range(10):
            print('nop')
    print('Cache is in directory', win32com.__gen_path__)
    d = {}
    for (clsid, (typelibCLSID, lcid, major, minor)) in clsidToTypelib.items():
        d[typelibCLSID, lcid, major, minor] = None
    for (typelibCLSID, lcid, major, minor) in d.keys():
        mod = GetModuleForTypelib(typelibCLSID, lcid, major, minor)
        print(f'{mod.__doc__} - {typelibCLSID}')
__init__()

def usage():
    if False:
        return 10
    usageString = '\t  Usage: gencache [-q] [-d] [-r]\n\n\t\t\t -q         - Quiet\n\t\t\t -d         - Dump the cache (typelibrary description and filename).\n\t\t\t -r         - Rebuild the cache dictionary from the existing .py files\n\t'
    print(usageString)
    sys.exit(1)
if __name__ == '__main__':
    import getopt
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'qrd')
    except getopt.error as message:
        print(message)
        usage()
    if len(sys.argv) == 1 or args:
        print(usage())
    verbose = 1
    for (opt, val) in opts:
        if opt == '-d':
            _Dump()
        if opt == '-r':
            Rebuild(verbose)
        if opt == '-q':
            verbose = 0