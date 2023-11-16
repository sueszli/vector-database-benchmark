import common, gcc, builtin
from b2.build import feature, toolset, type, action, generators
from b2.util.utility import *
toolset.register('darwin')
toolset.inherit_generators('darwin', [], 'gcc')
toolset.inherit_flags('darwin', 'gcc')
toolset.inherit_rules('darwin', 'gcc')

def init(version=None, command=None, options=None):
    if False:
        print('Hello World!')
    options = to_seq(options)
    condition = common.check_init_parameters('darwin', None, ('version', version))
    command = common.get_invocation_command('darwin', 'g++', command)
    common.handle_options('darwin', condition, command, options)
    gcc.init_link_flags('darwin', 'darwin', condition)
type.set_generated_target_suffix('SHARED_LIB', ['<toolset>darwin'], 'dylib')
type.register_suffixes('dylib', 'SHARED_LIB')
feature.feature('framework', [], ['free'])
toolset.flags('darwin.compile', 'OPTIONS', '<link>shared', ['-dynamic'])
toolset.flags('darwin.compile', 'OPTIONS', None, ['-Wno-long-double', '-no-cpp-precomp'])
toolset.flags('darwin.compile.c++', 'OPTIONS', None, ['-fcoalesce-templates'])
toolset.flags('darwin.link', 'FRAMEWORK', '<framework>')
action.register('darwin.compile.cpp', None, ['$(CONFIG_COMMAND) $(ST_OPTIONS) -L"$(LINKPATH)" -o "$(<)" "$(>)" "$(LIBRARIES)" -l$(FINDLIBS-SA) -l$(FINDLIBS-ST) -framework$(_)$(FRAMEWORK) $(OPTIONS)'])
action.register('darwin.link.dll', None, ['$(CONFIG_COMMAND) -dynamiclib -L"$(LINKPATH)" -o "$(<)" "$(>)" "$(LIBRARIES)" -l$(FINDLIBS-SA) -l$(FINDLIBS-ST) -framework$(_)$(FRAMEWORK) $(OPTIONS)'])

def darwin_archive(manager, targets, sources, properties):
    if False:
        return 10
    pass
action.register('darwin.archive', darwin_archive, ['ar -c -r -s $(ARFLAGS) "$(<:T)" "$(>:T)"'])