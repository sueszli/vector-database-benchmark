"""Generate a .py file from an OLE TypeLibrary file.


 This module is concerned only with the actual writing of
 a .py file.  It draws on the @build@ module, which builds
 the knowledge of a COM interface.

"""
usageHelp = ' \nUsage:\n\n  makepy.py [-i] [-v|q] [-h] [-u] [-o output_file] [-d] [typelib, ...]\n\n  -i    -- Show information for the specified typelib.\n\n  -v    -- Verbose output.\n\n  -q    -- Quiet output.\n\n  -h    -- Do not generate hidden methods.\n\n  -u    -- Convert all Unicode objects to strings.\n\n  -o    -- Create output in a specified output file.  If the path leading\n           to the file does not exist, any missing directories will be\n           created.\n           NOTE: -o cannot be used with -d.  This will generate an error.\n\n  -d    -- Generate the base code now and the class code on demand.\n           Recommended for large type libraries.\n\n  typelib -- A TLB, DLL, OCX or anything containing COM type information.\n             If a typelib is not specified, a window containing a textbox\n             will open from which you can select a registered type\n             library.\n\nExamples:\n\n  makepy.py -d\n\n    Presents a list of registered type libraries from which you can make\n    a selection.\n\n  makepy.py -d "Microsoft Excel 8.0 Object Library"\n\n    Generate support for the type library with the specified description\n    (in this case, the MS Excel object model).\n\n'
import importlib
import os
import sys
import pythoncom
from win32com.client import Dispatch, gencache, genpy, selecttlb
bForDemandDefault = 0

def usage():
    if False:
        for i in range(10):
            print('nop')
    sys.stderr.write(usageHelp)
    sys.exit(2)

def ShowInfo(spec):
    if False:
        while True:
            i = 10
    if not spec:
        tlbSpec = selecttlb.SelectTlb(excludeFlags=selecttlb.FLAG_HIDDEN)
        if tlbSpec is None:
            return
        try:
            tlb = pythoncom.LoadRegTypeLib(tlbSpec.clsid, tlbSpec.major, tlbSpec.minor, tlbSpec.lcid)
        except pythoncom.com_error:
            sys.stderr.write("Warning - could not load registered typelib '%s'\n" % tlbSpec.clsid)
            tlb = None
        infos = [(tlb, tlbSpec)]
    else:
        infos = GetTypeLibsForSpec(spec)
    for (tlb, tlbSpec) in infos:
        desc = tlbSpec.desc
        if desc is None:
            if tlb is None:
                desc = '<Could not load typelib %s>' % tlbSpec.dll
            else:
                desc = tlb.GetDocumentation(-1)[0]
        print(desc)
        print(' {}, lcid={}, major={}, minor={}'.format(tlbSpec.clsid, tlbSpec.lcid, tlbSpec.major, tlbSpec.minor))
        print(' >>> # Use these commands in Python code to auto generate .py support')
        print(' >>> from win32com.client import gencache')
        print(" >>> gencache.EnsureModule('{}', {}, {}, {})".format(tlbSpec.clsid, tlbSpec.lcid, tlbSpec.major, tlbSpec.minor))

class SimpleProgress(genpy.GeneratorProgress):
    """A simple progress class prints its output to stderr"""

    def __init__(self, verboseLevel):
        if False:
            for i in range(10):
                print('nop')
        self.verboseLevel = verboseLevel

    def Close(self):
        if False:
            return 10
        pass

    def Finished(self):
        if False:
            while True:
                i = 10
        if self.verboseLevel > 1:
            sys.stderr.write('Generation complete..\n')

    def SetDescription(self, desc, maxticks=None):
        if False:
            print('Hello World!')
        if self.verboseLevel:
            sys.stderr.write(desc + '\n')

    def Tick(self, desc=None):
        if False:
            print('Hello World!')
        pass

    def VerboseProgress(self, desc, verboseLevel=2):
        if False:
            print('Hello World!')
        if self.verboseLevel >= verboseLevel:
            sys.stderr.write(desc + '\n')

    def LogBeginGenerate(self, filename):
        if False:
            i = 10
            return i + 15
        self.VerboseProgress('Generating to %s' % filename, 1)

    def LogWarning(self, desc):
        if False:
            while True:
                i = 10
        self.VerboseProgress('WARNING: ' + desc, 1)

class GUIProgress(SimpleProgress):

    def __init__(self, verboseLevel):
        if False:
            return 10
        import pywin
        import win32ui
        SimpleProgress.__init__(self, verboseLevel)
        self.dialog = None

    def Close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dialog is not None:
            self.dialog.Close()
            self.dialog = None

    def Starting(self, tlb_desc):
        if False:
            print('Hello World!')
        SimpleProgress.Starting(self, tlb_desc)
        if self.dialog is None:
            from pywin.dialogs import status
            self.dialog = status.ThreadedStatusProgressDialog(tlb_desc)
        else:
            self.dialog.SetTitle(tlb_desc)

    def SetDescription(self, desc, maxticks=None):
        if False:
            print('Hello World!')
        self.dialog.SetText(desc)
        if maxticks:
            self.dialog.SetMaxTicks(maxticks)

    def Tick(self, desc=None):
        if False:
            for i in range(10):
                print('nop')
        self.dialog.Tick()
        if desc is not None:
            self.dialog.SetText(desc)

def GetTypeLibsForSpec(arg):
    if False:
        print('Hello World!')
    'Given an argument on the command line (either a file name, library\n    description, or ProgID of an object) return a list of actual typelibs\n    to use.'
    typelibs = []
    try:
        try:
            tlb = pythoncom.LoadTypeLib(arg)
            spec = selecttlb.TypelibSpec(None, 0, 0, 0)
            spec.FromTypelib(tlb, arg)
            typelibs.append((tlb, spec))
        except pythoncom.com_error:
            tlbs = selecttlb.FindTlbsWithDescription(arg)
            if len(tlbs) == 0:
                try:
                    ob = Dispatch(arg)
                    (tlb, index) = ob._oleobj_.GetTypeInfo().GetContainingTypeLib()
                    spec = selecttlb.TypelibSpec(None, 0, 0, 0)
                    spec.FromTypelib(tlb)
                    tlbs.append(spec)
                except pythoncom.com_error:
                    pass
            if len(tlbs) == 0:
                print("Could not locate a type library matching '%s'" % arg)
            for spec in tlbs:
                if spec.dll is None:
                    tlb = pythoncom.LoadRegTypeLib(spec.clsid, spec.major, spec.minor, spec.lcid)
                else:
                    tlb = pythoncom.LoadTypeLib(spec.dll)
                attr = tlb.GetLibAttr()
                spec.major = attr[3]
                spec.minor = attr[4]
                spec.lcid = attr[1]
                typelibs.append((tlb, spec))
        return typelibs
    except pythoncom.com_error:
        (t, v, tb) = sys.exc_info()
        sys.stderr.write(f"Unable to load type library from '{arg}' - {v}\n")
        tb = None
        sys.exit(1)

def GenerateFromTypeLibSpec(typelibInfo, file=None, verboseLevel=None, progressInstance=None, bUnicodeToString=None, bForDemand=bForDemandDefault, bBuildHidden=1):
    if False:
        print('Hello World!')
    assert bUnicodeToString is None, 'this is deprecated and will go away'
    if verboseLevel is None:
        verboseLevel = 0
    if bForDemand and file is not None:
        raise RuntimeError('You can only perform a demand-build when the output goes to the gen_py directory')
    if isinstance(typelibInfo, tuple):
        (typelibCLSID, lcid, major, minor) = typelibInfo
        tlb = pythoncom.LoadRegTypeLib(typelibCLSID, major, minor, lcid)
        spec = selecttlb.TypelibSpec(typelibCLSID, lcid, major, minor)
        spec.FromTypelib(tlb, str(typelibCLSID))
        typelibs = [(tlb, spec)]
    elif isinstance(typelibInfo, selecttlb.TypelibSpec):
        if typelibInfo.dll is None:
            tlb = pythoncom.LoadRegTypeLib(typelibInfo.clsid, typelibInfo.major, typelibInfo.minor, typelibInfo.lcid)
        else:
            tlb = pythoncom.LoadTypeLib(typelibInfo.dll)
        typelibs = [(tlb, typelibInfo)]
    elif hasattr(typelibInfo, 'GetLibAttr'):
        tla = typelibInfo.GetLibAttr()
        guid = tla[0]
        lcid = tla[1]
        major = tla[3]
        minor = tla[4]
        spec = selecttlb.TypelibSpec(guid, lcid, major, minor)
        typelibs = [(typelibInfo, spec)]
    else:
        typelibs = GetTypeLibsForSpec(typelibInfo)
    if progressInstance is None:
        progressInstance = SimpleProgress(verboseLevel)
    progress = progressInstance
    bToGenDir = file is None
    for (typelib, info) in typelibs:
        gen = genpy.Generator(typelib, info.dll, progress, bBuildHidden=bBuildHidden)
        if file is None:
            this_name = gencache.GetGeneratedFileName(info.clsid, info.lcid, info.major, info.minor)
            full_name = os.path.join(gencache.GetGeneratePath(), this_name)
            if bForDemand:
                try:
                    os.unlink(full_name + '.py')
                except OSError:
                    pass
                try:
                    os.unlink(full_name + '.pyc')
                except OSError:
                    pass
                try:
                    os.unlink(full_name + '.pyo')
                except OSError:
                    pass
                if not os.path.isdir(full_name):
                    os.mkdir(full_name)
                outputName = os.path.join(full_name, '__init__.py')
            else:
                outputName = full_name + '.py'
            fileUse = gen.open_writer(outputName)
            progress.LogBeginGenerate(outputName)
        else:
            fileUse = file
        worked = False
        try:
            gen.generate(fileUse, bForDemand)
            worked = True
        finally:
            if file is None:
                gen.finish_writer(outputName, fileUse, worked)
        importlib.invalidate_caches()
        if bToGenDir:
            progress.SetDescription('Importing module')
            gencache.AddModuleToCache(info.clsid, info.lcid, info.major, info.minor)
    progress.Close()

def GenerateChildFromTypeLibSpec(child, typelibInfo, verboseLevel=None, progressInstance=None, bUnicodeToString=None):
    if False:
        i = 10
        return i + 15
    assert bUnicodeToString is None, 'this is deprecated and will go away'
    if verboseLevel is None:
        verboseLevel = 0
    if isinstance(typelibInfo, tuple):
        (typelibCLSID, lcid, major, minor) = typelibInfo
        tlb = pythoncom.LoadRegTypeLib(typelibCLSID, major, minor, lcid)
    else:
        tlb = typelibInfo
        tla = typelibInfo.GetLibAttr()
        typelibCLSID = tla[0]
        lcid = tla[1]
        major = tla[3]
        minor = tla[4]
    spec = selecttlb.TypelibSpec(typelibCLSID, lcid, major, minor)
    spec.FromTypelib(tlb, str(typelibCLSID))
    typelibs = [(tlb, spec)]
    if progressInstance is None:
        progressInstance = SimpleProgress(verboseLevel)
    progress = progressInstance
    for (typelib, info) in typelibs:
        dir_name = gencache.GetGeneratedFileName(info.clsid, info.lcid, info.major, info.minor)
        dir_path_name = os.path.join(gencache.GetGeneratePath(), dir_name)
        progress.LogBeginGenerate(dir_path_name)
        gen = genpy.Generator(typelib, info.dll, progress)
        gen.generate_child(child, dir_path_name)
        progress.SetDescription('Importing module')
        importlib.invalidate_caches()
        __import__('win32com.gen_py.' + dir_name + '.' + child)
    progress.Close()

def main():
    if False:
        i = 10
        return i + 15
    import getopt
    hiddenSpec = 1
    outputName = None
    verboseLevel = 1
    doit = 1
    bForDemand = bForDemandDefault
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'vo:huiqd')
        for (o, v) in opts:
            if o == '-h':
                hiddenSpec = 0
            elif o == '-o':
                outputName = v
            elif o == '-v':
                verboseLevel = verboseLevel + 1
            elif o == '-q':
                verboseLevel = verboseLevel - 1
            elif o == '-i':
                if len(args) == 0:
                    ShowInfo(None)
                else:
                    for arg in args:
                        ShowInfo(arg)
                doit = 0
            elif o == '-d':
                bForDemand = not bForDemand
    except getopt.error as msg:
        sys.stderr.write(str(msg) + '\n')
        usage()
    if bForDemand and outputName is not None:
        sys.stderr.write('Can not use -d and -o together\n')
        usage()
    if not doit:
        return 0
    if len(args) == 0:
        rc = selecttlb.SelectTlb()
        if rc is None:
            sys.exit(1)
        args = [rc]
    if outputName is not None:
        path = os.path.dirname(outputName)
        if path != '' and (not os.path.exists(path)):
            os.makedirs(path)
        f = open(outputName, 'wt', encoding='mbcs')
    else:
        f = None
    for arg in args:
        GenerateFromTypeLibSpec(arg, f, verboseLevel=verboseLevel, bForDemand=bForDemand, bBuildHidden=hiddenSpec)
    if f:
        f.close()
if __name__ == '__main__':
    rc = main()
    if rc:
        sys.exit(rc)
    sys.exit(0)