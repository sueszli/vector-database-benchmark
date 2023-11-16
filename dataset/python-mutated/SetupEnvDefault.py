"""
Determine if and/or when an error/warning should be issued when there
are no versions of msvc installed.  If there is at least one version of
msvc installed, these routines do (almost) nothing.

Notes:
    * When msvc is the default compiler because there are no compilers
      installed, a build may fail due to the cl.exe command not being
      recognized.  Currently, there is no easy way to detect during
      msvc initialization if the default environment will be used later
      to build a program and/or library. There is no error/warning
      as there are legitimate SCons uses that do not require a c compiler.
    * An error is indicated by returning a non-empty tool list from the
      function register_iserror.
"""
import re
from ..common import debug
from . import Dispatcher
Dispatcher.register_modulename(__name__)

class _Data:
    separator = ';'
    need_init = True

    @classmethod
    def reset(cls):
        if False:
            for i in range(10):
                print('nop')
        debug('msvc default:init')
        cls.n_setup = 0
        cls.default_ismsvc = False
        cls.default_tools_re_list = []
        cls.msvc_tools_init = set()
        cls.msvc_tools = None
        cls.msvc_installed = False
        cls.msvc_nodefault = False
        cls.need_init = True

def _initialize(env, msvc_exists_func):
    if False:
        print('Hello World!')
    if _Data.need_init:
        _Data.reset()
        _Data.need_init = False
        _Data.msvc_installed = msvc_exists_func(env)
        debug('msvc default:msvc_installed=%s', _Data.msvc_installed)

def register_tool(env, tool, msvc_exists_func):
    if False:
        return 10
    if _Data.need_init:
        _initialize(env, msvc_exists_func)
    if _Data.msvc_installed:
        return None
    if not tool:
        return None
    if _Data.n_setup == 0:
        if tool not in _Data.msvc_tools_init:
            _Data.msvc_tools_init.add(tool)
            debug('msvc default:tool=%s, msvc_tools_init=%s', tool, _Data.msvc_tools_init)
        return None
    if tool not in _Data.msvc_tools:
        _Data.msvc_tools.add(tool)
        debug('msvc default:tool=%s, msvc_tools=%s', tool, _Data.msvc_tools)

def register_setup(env, msvc_exists_func):
    if False:
        i = 10
        return i + 15
    if _Data.need_init:
        _initialize(env, msvc_exists_func)
    _Data.n_setup += 1
    if not _Data.msvc_installed:
        _Data.msvc_tools = set(_Data.msvc_tools_init)
        if _Data.n_setup == 1:
            tool_list = env.get('TOOLS', None)
            if tool_list and tool_list[0] == 'default':
                if len(tool_list) > 1 and tool_list[1] in _Data.msvc_tools:
                    _Data.default_ismsvc = True
        _Data.msvc_nodefault = False
        debug('msvc default:n_setup=%d, msvc_installed=%s, default_ismsvc=%s', _Data.n_setup, _Data.msvc_installed, _Data.default_ismsvc)

def set_nodefault():
    if False:
        for i in range(10):
            print('nop')
    _Data.msvc_nodefault = True
    debug('msvc default:msvc_nodefault=%s', _Data.msvc_nodefault)

def register_iserror(env, tool, msvc_exists_func):
    if False:
        return 10
    register_tool(env, tool, msvc_exists_func)
    if _Data.msvc_installed:
        return None
    if not _Data.msvc_nodefault:
        return None
    tool_list = env.get('TOOLS', None)
    if not tool_list:
        return None
    debug('msvc default:n_setup=%s, default_ismsvc=%s, msvc_tools=%s, tool_list=%s', _Data.n_setup, _Data.default_ismsvc, _Data.msvc_tools, tool_list)
    if not _Data.default_ismsvc:
        tools_set = set(tool_list)
    else:
        if _Data.n_setup == 1:
            tools = _Data.separator.join(tool_list)
            tools_nchar = len(tools)
            debug('msvc default:add regex:nchar=%d, tools=%s', tools_nchar, tools)
            re_default_tools = re.compile(re.escape(tools))
            _Data.default_tools_re_list.insert(0, (tools_nchar, re_default_tools))
            return None
        tools = _Data.separator.join(tool_list)
        tools_nchar = len(tools)
        debug('msvc default:check tools:nchar=%d, tools=%s', tools_nchar, tools)
        (re_nchar_min, re_tools_min) = _Data.default_tools_re_list[-1]
        if tools_nchar >= re_nchar_min and re_tools_min.search(tools):
            for (re_nchar, re_default_tool) in _Data.default_tools_re_list:
                if tools_nchar < re_nchar:
                    continue
                tools = re_default_tool.sub('', tools).strip(_Data.separator)
                tools_nchar = len(tools)
                debug('msvc default:check tools:nchar=%d, tools=%s', tools_nchar, tools)
                if tools_nchar < re_nchar_min or not re_tools_min.search(tools):
                    break
        tools_set = {msvc_tool for msvc_tool in tools.split(_Data.separator) if msvc_tool}
    debug('msvc default:tools=%s', tools_set)
    if not tools_set:
        return None
    tools_found = _Data.msvc_tools.intersection(tools_set)
    debug('msvc default:tools_exist=%s', tools_found)
    if not tools_found:
        return None
    tools_found_list = []
    seen_tool = set()
    for tool in tool_list:
        if tool not in seen_tool:
            seen_tool.add(tool)
            if tool in tools_found:
                tools_found_list.append(tool)
    return tools_found_list

def reset():
    if False:
        print('Hello World!')
    debug('')
    _Data.reset()