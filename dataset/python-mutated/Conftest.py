"""SCons.Conftest

Autoconf-like configuration support; low level implementation of tests.
"""
import re
LogInputFiles = 1
LogErrorMessages = 1

def CheckBuilder(context, text=None, language=None):
    if False:
        print('Hello World!')
    '\n    Configure check to see if the compiler works.\n    Note that this uses the current value of compiler and linker flags, make\n    sure $CFLAGS, $CPPFLAGS and $LIBS are set correctly.\n    "language" should be "C" or "C++" and is used to select the compiler.\n    Default is "C".\n    "text" may be used to specify the code to be build.\n    Returns an empty string for success, an error message for failure.\n    '
    (lang, suffix, msg) = _lang2suffix(language)
    if msg:
        context.Display('%s\n' % msg)
        return msg
    if not text:
        text = '\nint main(void) {\n    return 0;\n}\n'
    context.Display('Checking if building a %s file works... ' % lang)
    ret = context.BuildProg(text, suffix)
    _YesNoResult(context, ret, None, text)
    return ret

def CheckCC(context):
    if False:
        for i in range(10):
            print('nop')
    '\n    Configure check for a working C compiler.\n\n    This checks whether the C compiler, as defined in the $CC construction\n    variable, can compile a C source file. It uses the current $CCCOM value\n    too, so that it can test against non working flags.\n\n    '
    context.Display('Checking whether the C compiler works... ')
    text = '\nint main(void)\n{\n    return 0;\n}\n'
    ret = _check_empty_program(context, 'CC', text, 'C')
    _YesNoResult(context, ret, None, text)
    return ret

def CheckSHCC(context):
    if False:
        print('Hello World!')
    '\n    Configure check for a working shared C compiler.\n\n    This checks whether the C compiler, as defined in the $SHCC construction\n    variable, can compile a C source file. It uses the current $SHCCCOM value\n    too, so that it can test against non working flags.\n\n    '
    context.Display('Checking whether the (shared) C compiler works... ')
    text = '\nint foo(void)\n{\n    return 0;\n}\n'
    ret = _check_empty_program(context, 'SHCC', text, 'C', use_shared=True)
    _YesNoResult(context, ret, None, text)
    return ret

def CheckCXX(context):
    if False:
        for i in range(10):
            print('nop')
    '\n    Configure check for a working CXX compiler.\n\n    This checks whether the CXX compiler, as defined in the $CXX construction\n    variable, can compile a CXX source file. It uses the current $CXXCOM value\n    too, so that it can test against non working flags.\n\n    '
    context.Display('Checking whether the C++ compiler works... ')
    text = '\nint main(void)\n{\n    return 0;\n}\n'
    ret = _check_empty_program(context, 'CXX', text, 'C++')
    _YesNoResult(context, ret, None, text)
    return ret

def CheckSHCXX(context):
    if False:
        while True:
            i = 10
    '\n    Configure check for a working shared CXX compiler.\n\n    This checks whether the CXX compiler, as defined in the $SHCXX construction\n    variable, can compile a CXX source file. It uses the current $SHCXXCOM value\n    too, so that it can test against non working flags.\n\n    '
    context.Display('Checking whether the (shared) C++ compiler works... ')
    text = '\nint main(void)\n{\n    return 0;\n}\n'
    ret = _check_empty_program(context, 'SHCXX', text, 'C++', use_shared=True)
    _YesNoResult(context, ret, None, text)
    return ret

def _check_empty_program(context, comp, text, language, use_shared=False):
    if False:
        i = 10
        return i + 15
    'Return 0 on success, 1 otherwise.'
    if comp not in context.env or not context.env[comp]:
        return 1
    (lang, suffix, msg) = _lang2suffix(language)
    if msg:
        return 1
    if use_shared:
        return context.CompileSharedObject(text, suffix)
    else:
        return context.CompileProg(text, suffix)

def CheckFunc(context, function_name, header=None, language=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Configure check for a function "function_name".\n    "language" should be "C" or "C++" and is used to select the compiler.\n    Default is "C".\n    Optional "header" can be defined to define a function prototype, include a\n    header file or anything else that comes before main().\n    Sets HAVE_function_name in context.havedict according to the result.\n    Note that this uses the current value of compiler and linker flags, make\n    sure $CFLAGS, $CPPFLAGS and $LIBS are set correctly.\n    Returns an empty string for success, an error message for failure.\n    '
    if context.headerfilename:
        includetext = '#include "%s"' % context.headerfilename
    else:
        includetext = ''
    if not header:
        header = '\n#ifdef __cplusplus\nextern "C"\n#endif\nchar %s();' % function_name
    (lang, suffix, msg) = _lang2suffix(language)
    if msg:
        context.Display('Cannot check for %s(): %s\n' % (function_name, msg))
        return msg
    text = '\n%(include)s\n#include <assert.h>\n%(hdr)s\n\n#if _MSC_VER && !__INTEL_COMPILER\n    #pragma function(%(name)s)\n#endif\n\nint main(void) {\n#if defined (__stub_%(name)s) || defined (__stub___%(name)s)\n  fail fail fail\n#else\n  %(name)s();\n#endif\n\n  return 0;\n}\n' % {'name': function_name, 'include': includetext, 'hdr': header}
    context.Display('Checking for %s function %s()... ' % (lang, function_name))
    ret = context.BuildProg(text, suffix)
    _YesNoResult(context, ret, 'HAVE_' + function_name, text, "Define to 1 if the system has the function `%s'." % function_name)
    return ret

def CheckHeader(context, header_name, header=None, language=None, include_quotes=None):
    if False:
        i = 10
        return i + 15
    '\n    Configure check for a C or C++ header file "header_name".\n    Optional "header" can be defined to do something before including the\n    header file (unusual, supported for consistency).\n    "language" should be "C" or "C++" and is used to select the compiler.\n    Default is "C".\n    Sets HAVE_header_name in context.havedict according to the result.\n    Note that this uses the current value of compiler and linker flags, make\n    sure $CFLAGS and $CPPFLAGS are set correctly.\n    Returns an empty string for success, an error message for failure.\n    '
    if context.headerfilename:
        includetext = '#include "%s"\n' % context.headerfilename
    else:
        includetext = ''
    if not header:
        header = ''
    (lang, suffix, msg) = _lang2suffix(language)
    if msg:
        context.Display('Cannot check for header file %s: %s\n' % (header_name, msg))
        return msg
    if not include_quotes:
        include_quotes = '<>'
    text = '%s%s\n#include %s%s%s\n\n' % (includetext, header, include_quotes[0], header_name, include_quotes[1])
    context.Display('Checking for %s header file %s... ' % (lang, header_name))
    ret = context.CompileProg(text, suffix)
    _YesNoResult(context, ret, 'HAVE_' + header_name, text, 'Define to 1 if you have the <%s> header file.' % header_name)
    return ret

def CheckType(context, type_name, fallback=None, header=None, language=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Configure check for a C or C++ type "type_name".\n    Optional "header" can be defined to include a header file.\n    "language" should be "C" or "C++" and is used to select the compiler.\n    Default is "C".\n    Sets HAVE_type_name in context.havedict according to the result.\n    Note that this uses the current value of compiler and linker flags, make\n    sure $CFLAGS, $CPPFLAGS and $LIBS are set correctly.\n    Returns an empty string for success, an error message for failure.\n    '
    if context.headerfilename:
        includetext = '#include "%s"' % context.headerfilename
    else:
        includetext = ''
    if not header:
        header = ''
    (lang, suffix, msg) = _lang2suffix(language)
    if msg:
        context.Display('Cannot check for %s type: %s\n' % (type_name, msg))
        return msg
    text = '\n%(include)s\n%(header)s\n\nint main(void) {\n  if ((%(name)s *) 0)\n    return 0;\n  if (sizeof (%(name)s))\n    return 0;\n}\n' % {'include': includetext, 'header': header, 'name': type_name}
    context.Display('Checking for %s type %s... ' % (lang, type_name))
    ret = context.BuildProg(text, suffix)
    _YesNoResult(context, ret, 'HAVE_' + type_name, text, "Define to 1 if the system has the type `%s'." % type_name)
    if ret and fallback and context.headerfilename:
        f = open(context.headerfilename, 'a')
        f.write('typedef %s %s;\n' % (fallback, type_name))
        f.close()
    return ret

def CheckTypeSize(context, type_name, header=None, language=None, expect=None):
    if False:
        i = 10
        return i + 15
    "This check can be used to get the size of a given type, or to check whether\n    the type is of expected size.\n\n    Arguments:\n        - type : str\n            the type to check\n        - includes : sequence\n            list of headers to include in the test code before testing the type\n        - language : str\n            'C' or 'C++'\n        - expect : int\n            if given, will test wether the type has the given number of bytes.\n            If not given, will automatically find the size.\n\n        Returns:\n            status : int\n                0 if the check failed, or the found size of the type if the check succeeded."
    if context.headerfilename:
        includetext = '#include "%s"' % context.headerfilename
    else:
        includetext = ''
    if not header:
        header = ''
    (lang, suffix, msg) = _lang2suffix(language)
    if msg:
        context.Display('Cannot check for %s type: %s\n' % (type_name, msg))
        return msg
    src = includetext + header
    if expect is not None:
        context.Display('Checking %s is %d bytes... ' % (type_name, expect))
        src = src + '\ntypedef %s scons_check_type;\n\nint main(void)\n{\n    static int test_array[1 - 2 * !(((long int) (sizeof(scons_check_type))) == %d)];\n    test_array[0] = 0;\n\n    return 0;\n}\n'
        st = context.CompileProg(src % (type_name, expect), suffix)
        if not st:
            context.Display('yes\n')
            _Have(context, 'SIZEOF_%s' % type_name, expect, "The size of `%s', as computed by sizeof." % type_name)
            return expect
        else:
            context.Display('no\n')
            _LogFailed(context, src, st)
            return 0
    else:
        context.Message('Checking size of %s ... ' % type_name)
        src = src + '\n#include <stdlib.h>\n#include <stdio.h>\nint main(void) {\n    printf("%d", (int)sizeof(' + type_name + '));\n    return 0;\n}\n    '
        (st, out) = context.RunProg(src, suffix)
        try:
            size = int(out)
        except ValueError:
            st = 1
            size = 0
        if not st:
            context.Display('yes\n')
            _Have(context, 'SIZEOF_%s' % type_name, size, "The size of `%s', as computed by sizeof." % type_name)
            return size
        else:
            context.Display('no\n')
            _LogFailed(context, src, st)
            return 0
    return 0

def CheckDeclaration(context, symbol, includes=None, language=None):
    if False:
        while True:
            i = 10
    'Checks whether symbol is declared.\n\n    Use the same test as autoconf, that is test whether the symbol is defined\n    as a macro or can be used as an r-value.\n\n    Arguments:\n        symbol : str\n            the symbol to check\n        includes : str\n            Optional "header" can be defined to include a header file.\n        language : str\n            only C and C++ supported.\n\n    Returns:\n        status : bool\n            True if the check failed, False if succeeded.'
    if context.headerfilename:
        includetext = '#include "%s"' % context.headerfilename
    else:
        includetext = ''
    if not includes:
        includes = ''
    (lang, suffix, msg) = _lang2suffix(language)
    if msg:
        context.Display('Cannot check for declaration %s: %s\n' % (symbol, msg))
        return msg
    src = includetext + includes
    context.Display('Checking whether %s is declared... ' % symbol)
    src = src + '\nint main(void)\n{\n#ifndef %s\n    (void) %s;\n#endif\n    ;\n    return 0;\n}\n' % (symbol, symbol)
    st = context.CompileProg(src, suffix)
    _YesNoResult(context, st, 'HAVE_DECL_' + symbol, src, 'Set to 1 if %s is defined.' % symbol)
    return st

def CheckLib(context, libs, func_name=None, header=None, extra_libs=None, call=None, language=None, autoadd=1, append=True):
    if False:
        i = 10
        return i + 15
    '\n    Configure check for a C or C++ libraries "libs".  Searches through\n    the list of libraries, until one is found where the test succeeds.\n    Tests if "func_name" or "call" exists in the library.  Note: if it exists\n    in another library the test succeeds anyway!\n    Optional "header" can be defined to include a header file.  If not given a\n    default prototype for "func_name" is added.\n    Optional "extra_libs" is a list of library names to be added after\n    "lib_name" in the build command.  To be used for libraries that "lib_name"\n    depends on.\n    Optional "call" replaces the call to "func_name" in the test code.  It must\n    consist of complete C statements, including a trailing ";".\n    Both "func_name" and "call" arguments are optional, and in that case, just\n    linking against the libs is tested.\n    "language" should be "C" or "C++" and is used to select the compiler.\n    Default is "C".\n    Note that this uses the current value of compiler and linker flags, make\n    sure $CFLAGS, $CPPFLAGS and $LIBS are set correctly.\n    Returns an empty string for success, an error message for failure.\n    '
    if context.headerfilename:
        includetext = '#include "%s"' % context.headerfilename
    else:
        includetext = ''
    if not header:
        header = ''
    text = '\n%s\n%s' % (includetext, header)
    if func_name and func_name != 'main':
        if not header:
            text = text + '\n#ifdef __cplusplus\nextern "C"\n#endif\nchar %s();\n' % func_name
        if not call:
            call = '%s();' % func_name
    text = text + '\nint\nmain() {\n  %s\nreturn 0;\n}\n' % (call or '')
    if call:
        i = call.find('\n')
        if i > 0:
            calltext = call[:i] + '..'
        elif call[-1] == ';':
            calltext = call[:-1]
        else:
            calltext = call
    for lib_name in libs:
        (lang, suffix, msg) = _lang2suffix(language)
        if msg:
            context.Display('Cannot check for library %s: %s\n' % (lib_name, msg))
            return msg
        if call:
            context.Display('Checking for %s in %s library %s... ' % (calltext, lang, lib_name))
        else:
            context.Display('Checking for %s library %s... ' % (lang, lib_name))
        if lib_name:
            l = [lib_name]
            if extra_libs:
                l.extend(extra_libs)
            if append:
                oldLIBS = context.AppendLIBS(l)
            else:
                oldLIBS = context.PrependLIBS(l)
            sym = 'HAVE_LIB' + lib_name
        else:
            oldLIBS = -1
            sym = None
        ret = context.BuildProg(text, suffix)
        _YesNoResult(context, ret, sym, text, "Define to 1 if you have the `%s' library." % lib_name)
        if oldLIBS != -1 and (ret or not autoadd):
            context.SetLIBS(oldLIBS)
        if not ret:
            return ret
    return ret

def CheckProg(context, prog_name):
    if False:
        print('Hello World!')
    '\n    Configure check for a specific program.\n\n    Check whether program prog_name exists in path.  If it is found,\n    returns the path for it, otherwise returns None.\n    '
    context.Display('Checking whether %s program exists...' % prog_name)
    path = context.env.WhereIs(prog_name)
    if path:
        context.Display(path + '\n')
    else:
        context.Display('no\n')
    return path

def _YesNoResult(context, ret, key, text, comment=None):
    if False:
        print('Hello World!')
    '\n    Handle the result of a test with a "yes" or "no" result.\n\n    :Parameters:\n      - `ret` is the return value: empty if OK, error message when not.\n      - `key` is the name of the symbol to be defined (HAVE_foo).\n      - `text` is the source code of the program used for testing.\n      - `comment` is the C comment to add above the line defining the symbol (the comment is automatically put inside a /\\* \\*/). If None, no comment is added.\n    '
    if key:
        _Have(context, key, not ret, comment)
    if ret:
        context.Display('no\n')
        _LogFailed(context, text, ret)
    else:
        context.Display('yes\n')

def _Have(context, key, have, comment=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Store result of a test in context.havedict and context.headerfilename.\n\n    :Parameters:\n      - `key` - is a "HAVE_abc" name.  It is turned into all CAPITALS and non-alphanumerics are replaced by an underscore.\n      - `have`   - value as it should appear in the header file, include quotes when desired and escape special characters!\n      - `comment` is the C comment to add above the line defining the symbol (the comment is automatically put inside a /\\* \\*/). If None, no comment is added.\n\n\n    The value of "have" can be:\n      - 1      - Feature is defined, add "#define key".\n      - 0      - Feature is not defined, add "/\\* #undef key \\*/". Adding "undef" is what autoconf does.  Not useful for the compiler, but it shows that the test was done.\n      - number - Feature is defined to this number "#define key have". Doesn\'t work for 0 or 1, use a string then.\n      - string - Feature is defined to this string "#define key have".\n\n\n    '
    key_up = key.upper()
    key_up = re.sub('[^A-Z0-9_]', '_', key_up)
    context.havedict[key_up] = have
    if have == 1:
        line = '#define %s 1\n' % key_up
    elif have == 0:
        line = '/* #undef %s */\n' % key_up
    elif isinstance(have, int):
        line = '#define %s %d\n' % (key_up, have)
    else:
        line = '#define %s %s\n' % (key_up, str(have))
    if comment is not None:
        lines = '\n/* %s */\n' % comment + line
    else:
        lines = '\n' + line
    if context.headerfilename:
        f = open(context.headerfilename, 'a')
        f.write(lines)
        f.close()
    elif hasattr(context, 'config_h'):
        context.config_h = context.config_h + lines

def _LogFailed(context, text, msg):
    if False:
        i = 10
        return i + 15
    '\n    Write to the log about a failed program.\n    Add line numbers, so that error messages can be understood.\n    '
    if LogInputFiles:
        context.Log('Failed program was:\n')
        lines = text.split('\n')
        if len(lines) and lines[-1] == '':
            lines = lines[:-1]
        n = 1
        for line in lines:
            context.Log('%d: %s\n' % (n, line))
            n = n + 1
    if LogErrorMessages:
        context.Log('Error message: %s\n' % msg)

def _lang2suffix(lang):
    if False:
        while True:
            i = 10
    '\n    Convert a language name to a suffix.\n    When "lang" is empty or None C is assumed.\n    Returns a tuple (lang, suffix, None) when it works.\n    For an unrecognized language returns (None, None, msg).\n\n    Where:\n      - lang   = the unified language name\n      - suffix = the suffix, including the leading dot\n      - msg    = an error message\n    '
    if not lang or lang in ['C', 'c']:
        return ('C', '.c', None)
    if lang in ['c++', 'C++', 'cpp', 'CXX', 'cxx']:
        return ('C++', '.cpp', None)
    return (None, None, 'Unsupported language: %s' % lang)