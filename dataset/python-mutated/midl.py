from b2.build import scanner, type
from b2.build.toolset import flags
from b2.build.feature import feature
from b2.manager import get_manager
from b2.tools import builtin, common
from b2.util import regex, utility

def init():
    if False:
        for i in range(10):
            print('nop')
    pass
type.register('IDL', ['idl'])
type.register('MSTYPELIB', ['tlb'], 'H')

class MidlScanner(scanner.Scanner):

    def __init__(self, includes=[]):
        if False:
            for i in range(10):
                print('nop')
        scanner.Scanner.__init__(self)
        self.includes = includes
        re_strings = '[ \t]*"([^"]*)"([ \t]*,[ \t]*"([^"]*)")*[ \t]*'
        self.re_import = 'import' + re_strings + '[ \t]*;'
        self.re_importlib = 'importlib[ \t]*[(]' + re_strings + '[)][ \t]*;'
        self.re_include_angle = '#[ \t]*include[ \t]*<(.*)>'
        self.re_include_quoted = '#[ \t]*include[ \t]*"(.*)"'

    def pattern():
        if False:
            return 10
        return '((#[ \t]*include|import(lib)?).+(<(.*)>|"(.*)").+)'

    def process(self, target, matches, binding):
        if False:
            for i in range(10):
                print('nop')
        included_angle = regex.transform(matches, self.re_include_angle)
        included_quoted = regex.transform(matches, self.re_include_quoted)
        imported = regex.transform(matches, self.re_import, [1, 3])
        imported_tlbs = regex.transform(matches, self.re_importlib, [1, 3])
        g = bjam.call('get-target-variable', target, 'HDRGRIST')[0]
        b = os.path.normpath(os.path.dirname(binding))
        g2 = g + '#' + b
        g = '<' + g + '>'
        g2 = '<' + g2 + '>'
        included_angle = [g + x for x in included_angle]
        included_quoted = [g + x for x in included_quoted]
        imported = [g + x for x in imported]
        imported_tlbs = [g + x for x in imported_tlbs]
        all = included_angle + included_quoted + imported
        bjam.call('INCLUDES', [target], all)
        bjam.call('DEPENDS', [target], imported_tlbs)
        bjam.call('NOCARE', all + imported_tlbs)
        engine.set_target_variable(included_angle, 'SEARCH', [utility.get_value(inc) for inc in self.includes])
        engine.set_target_variable(included_quoted, 'SEARCH', [utility.get_value(inc) for inc in self.includes])
        engine.set_target_variable(imported, 'SEARCH', [utility.get_value(inc) for inc in self.includes])
        engine.set_target_variable(imported_tlbs, 'SEARCH', [utility.get_value(inc) for inc in self.includes])
        get_manager().scanners().propagate(type.get_scanner('CPP', PropertySet(self.includes)), included_angle + included_quoted)
        get_manager().scanners().propagate(self, imported)
scanner.register(MidlScanner, 'include')
type.set_scanner('IDL', MidlScanner)
feature('midl-stubless-proxy', ['yes', 'no'], ['propagated'])
feature('midl-robust', ['yes', 'no'], ['propagated'])
flags('midl.compile.idl', 'MIDLFLAGS', ['<midl-stubless-proxy>yes'], ['/Oicf'])
flags('midl.compile.idl', 'MIDLFLAGS', ['<midl-stubless-proxy>no'], ['/Oic'])
flags('midl.compile.idl', 'MIDLFLAGS', ['<midl-robust>yes'], ['/robust'])
flags('midl.compile.idl', 'MIDLFLAGS', ['<midl-robust>no'], ['/no_robust'])
architecture_x86 = ['<architecture>', '<architecture>x86']
address_model_32 = ['<address-model>', '<address-model>32']
address_model_64 = ['<address-model>', '<address-model>64']
flags('midl.compile.idl', 'MIDLFLAGS', [ar + '/' + m for ar in architecture_x86 for m in address_model_32], ['/win32'])
flags('midl.compile.idl', 'MIDLFLAGS', [ar + '/<address-model>64' for ar in architecture_x86], ['/x64'])
flags('midl.compile.idl', 'MIDLFLAGS', ['<architecture>ia64/' + m for m in address_model_64], ['/ia64'])
flags('midl.compile.idl', 'DEFINES', [], ['<define>'])
flags('midl.compile.idl', 'UNDEFS', [], ['<undef>'])
flags('midl.compile.idl', 'INCLUDES', [], ['<include>'])
builtin.register_c_compiler('midl.compile.idl', ['IDL'], ['MSTYPELIB', 'H', 'C(%_i)', 'C(%_proxy)', 'C(%_dlldata)'], [])
get_manager().engine().register_action('midl.compile.idl', 'midl /nologo @"@($(<[1]:W).rsp:E=\n"$(>:W)"\n-D$(DEFINES)\n"-I$(INCLUDES)"\n-U$(UNDEFS)\n$(MIDLFLAGS)\n/tlb "$(<[1]:W)"\n/h "$(<[2]:W)"\n/iid "$(<[3]:W)"\n/proxy "$(<[4]:W)"\n/dlldata "$(<[5]:W)")"\n{touch} "$(<[4]:W)"\n{touch} "$(<[5]:W)"'.format(touch=common.file_creation_command()))