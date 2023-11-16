from waflib import Utils
from waflib.Configure import conf

@conf
def d_platform_flags(self):
    if False:
        i = 10
        return i + 15
    v = self.env
    if not v.DEST_OS:
        v.DEST_OS = Utils.unversioned_sys_platform()
    binfmt = Utils.destos_to_binfmt(self.env.DEST_OS)
    if binfmt == 'pe':
        v.dprogram_PATTERN = '%s.exe'
        v.dshlib_PATTERN = 'lib%s.dll'
        v.dstlib_PATTERN = 'lib%s.a'
    elif binfmt == 'mac-o':
        v.dprogram_PATTERN = '%s'
        v.dshlib_PATTERN = 'lib%s.dylib'
        v.dstlib_PATTERN = 'lib%s.a'
    else:
        v.dprogram_PATTERN = '%s'
        v.dshlib_PATTERN = 'lib%s.so'
        v.dstlib_PATTERN = 'lib%s.a'
DLIB = '\nversion(D_Version2) {\n\timport std.stdio;\n\tint main() {\n\t\twritefln("phobos2");\n\t\treturn 0;\n\t}\n} else {\n\tversion(Tango) {\n\t\timport tango.stdc.stdio;\n\t\tint main() {\n\t\t\tprintf("tango");\n\t\t\treturn 0;\n\t\t}\n\t} else {\n\t\timport std.stdio;\n\t\tint main() {\n\t\t\twritefln("phobos1");\n\t\t\treturn 0;\n\t\t}\n\t}\n}\n'

@conf
def check_dlibrary(self, execute=True):
    if False:
        for i in range(10):
            print('nop')
    ret = self.check_cc(features='d dprogram', fragment=DLIB, compile_filename='test.d', execute=execute, define_ret=True)
    if execute:
        self.env.DLIBRARY = ret.strip()