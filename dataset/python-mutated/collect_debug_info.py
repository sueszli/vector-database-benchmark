import BoostBuild
import os
import re
import sys

def collectDebugInfo():
    if False:
        for i in range(10):
            print('nop')
    t = _init()
    global tag
    tag = 'Python version'
    try:
        _info(sys.version)
    except:
        _info_exc()
    tag = 'Python platform'
    try:
        _info(sys.platform)
    except:
        _info_exc()
    tag = 'Boost Jam/Build version'
    try:
        _infoX(_getJamVersionInfo(t))
    except:
        _info_exc()
    t.fail_test(1, dump_difference=False, dump_stdio=False, dump_stack=False)
varSeparator = '###$^%~~~'

def _collect(results, prefix, name, t):
    if False:
        return 10
    results.append('%s - %s - os.getenv(): %r' % (prefix, name, os.getenv(name)))
    results.append('%s - %s - os.environ.get(): %r' % (prefix, name, os.environ.get(name)))
    external_values = _getExternalValues(t, name)
    results.append('%s - %s - external: %r' % (prefix, name, external_values[name]))

def _collectDebugInfo_environ(t):
    if False:
        i = 10
        return i + 15
    dummyVars = ['WOOF_WOOFIE_%d' % x for x in xrange(4)]
    global tag
    tag = 'XXX in os.environ'
    try:

        def f(name):
            if False:
                while True:
                    i = 10
            return '%s: %s' % (name, name in os.environ)
        _infoX((f(x) for x in dummyVars))
    except:
        _info_exc()
    tag = 'os.environ[XXX]'
    try:

        def f(name):
            if False:
                for i in range(10):
                    print('nop')
            try:
                result = os.environ[name]
            except:
                result = _str_exc()
            return '%s: %r' % (name, result)
        _infoX((f(x) for x in dummyVars))
    except:
        _info_exc()
    tag = 'os.environ.get(XXX)'
    try:

        def f(name):
            if False:
                return 10
            return '%s: %r' % (name, os.environ.get(name))
        _infoX((f(x) for x in dummyVars))
    except:
        _info_exc()
    tag = 'os.getenv(XXX)'
    try:

        def f(name):
            if False:
                while True:
                    i = 10
            return '%s: %r' % (name, os.getenv(name))
        _infoX((f(x) for x in dummyVars))
    except:
        _info_exc()
    name = dummyVars[0]
    value = 'foo'
    tag = 'os.putenv(%s) to %r' % (name, value)
    try:
        results = []
        _collect(results, 'before', name, t)
        os.putenv(name, value)
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()
    name = dummyVars[1]
    value = 'bar'
    tag = 'os.environ[%s] to %r' % (name, value)
    try:
        results = []
        _collect(results, 'before', name, t)
        os.environ[name] = value
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()
    name = dummyVars[1]
    value = 'baz'
    tag = 'os.putenv(%s) to %r' % (name, value)
    try:
        results = []
        _collect(results, 'before', name, t)
        os.putenv(name, value)
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()
    name = dummyVars[1]
    value = ''
    tag = 'os.putenv(%s) to %r' % (name, value)
    try:
        results = []
        _collect(results, 'before', name, t)
        os.putenv(name, value)
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()
    name = dummyVars[2]
    value = 'foo'
    tag = 'os.unsetenv(%s) from %r' % (name, value)
    try:
        results = []
        os.environ[name] = value
        _collect(results, 'before', name, t)
        os.unsetenv(name)
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()
    name = dummyVars[2]
    value = 'foo'
    tag = 'del os.environ[%s] from %r' % (name, value)
    try:
        results = []
        os.environ[name] = value
        _collect(results, 'before', name, t)
        del os.environ[name]
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()
    name = dummyVars[2]
    value = 'foo'
    tag = 'os.environ.pop(%s) from %r' % (name, value)
    try:
        results = []
        os.environ[name] = value
        _collect(results, 'before', name, t)
        os.environ.pop(name)
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()
    name = dummyVars[2]
    value1 = 'foo'
    value2 = ''
    tag = 'os.environ[%s] to %r from %r' % (name, value2, value1)
    try:
        results = []
        os.environ[name] = value1
        _collect(results, 'before', name, t)
        os.environ[name] = value2
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()
    name = dummyVars[3]
    value = '""'
    tag = 'os.environ[%s] to %r' % (name, value)
    try:
        results = []
        _collect(results, 'before', name, t)
        os.environ[name] = value
        _collect(results, 'after', name, t)
        _infoX(results)
    except:
        _info_exc()

def _getExternalValues(t, *args):
    if False:
        print('Hello World!')
    t.run_build_system(['---var-name=%s' % x for x in args])
    result = dict()
    for x in args:
        m = re.search("^\\*\\*\\*ENV\\*\\*\\* %s: '(.*)' \\*\\*\\*$" % x, t.stdout(), re.MULTILINE)
        if m:
            result[x] = m.group(1)
        else:
            result[x] = None
    return result

def _getJamVersionInfo(t):
    if False:
        print('Hello World!')
    result = []
    t.run_build_system(['---version'])
    for m in re.finditer('^\\*\\*\\*VAR\\*\\*\\* ([^:]*): (.*)\\*\\*\\*$', t.stdout(), re.MULTILINE):
        name = m.group(1)
        value = m.group(2)
        if not value:
            value = []
        elif value[-1] == ' ':
            value = value[:-1].split(varSeparator)
        else:
            value = "!!!INVALID!!! - '%s'" % value
        result.append('%s = %s' % (name, value))
    result.append('')
    t.run_build_system(['-v'])
    result.append("--- output for 'bjam -v' ---")
    result.append(t.stdout())
    t.run_build_system(['--version'], status=1)
    result.append("--- output for 'bjam --version' ---")
    result.append(t.stdout())
    return result

def _init():
    if False:
        while True:
            i = 10
    toolsetName = '__myDummyToolset__'
    t = BoostBuild.Tester(['toolset=%s' % toolsetName], pass_toolset=False, use_test_config=False)
    t.write(toolsetName + '.jam', 'import feature ;\nfeature.extend toolset : %s ;\nrule init ( ) { }\n' % toolsetName)
    t.write(toolsetName + '.py', "from b2.build import feature\nfeature.extend('toolset', ['%s'])\ndef init(): pass\n" % toolsetName)
    t.write('jamroot.jam', 'import os ;\n.argv = [ modules.peek : ARGV ] ;\nlocal names = [ MATCH ^---var-name=(.*) : $(.argv) ] ;\nfor x in $(names)\n{\n    value = [ os.environ $(x) ] ;\n    ECHO ***ENV*** $(x): \'$(value)\' *** ;\n}\nif ---version in $(.argv)\n{\n    for x in JAMVERSION JAM_VERSION JAMUNAME JAM_TIMESTAMP_RESOLUTION OS\n    {\n        v = [ modules.peek : $(x) ] ;\n        ECHO ***VAR*** $(x): "$(v:J=%s)" *** ;\n    }\n}\n' % varSeparator)
    return t

def _info(*values):
    if False:
        return 10
    values = list(values) + ['']
    BoostBuild.annotation(tag, '\n'.join((str(x) for x in values)))

def _infoX(values):
    if False:
        print('Hello World!')
    _info(*values)

def _info_exc():
    if False:
        for i in range(10):
            print('nop')
    _info(_str_exc())

def _str_exc():
    if False:
        while True:
            i = 10
    (exc_type, exc_value) = sys.exc_info()[0:2]
    if exc_type is None:
        exc_type_name = 'None'
    else:
        exc_type_name = exc_type.__name__
    return '*** EXCEPTION *** %s - %s ***' % (exc_type_name, exc_value)
collectDebugInfo()