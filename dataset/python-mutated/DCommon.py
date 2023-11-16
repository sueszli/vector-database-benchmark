"""SCons.Tool.DCommon

Common code for the various D tools.

Coded by Russel Winder (russel@winder.org.uk)
2012-09-06
"""
import os.path

def isD(env, source):
    if False:
        while True:
            i = 10
    if not source:
        return 0
    for s in source:
        if s.sources:
            ext = os.path.splitext(str(s.sources[0]))[1]
            if ext == '.d':
                return 1
    return 0

def addDPATHToEnv(env, executable):
    if False:
        print('Hello World!')
    dPath = env.WhereIs(executable)
    if dPath:
        phobosDir = dPath[:dPath.rindex(executable)] + '/../src/phobos'
        if os.path.isdir(phobosDir):
            env.Append(DPATH=[phobosDir])

def allAtOnceEmitter(target, source, env):
    if False:
        for i in range(10):
            print('nop')
    if env['DC'] in ('ldc2', 'dmd'):
        env.SideEffect(str(target[0]) + '.o', target[0])
        env.Clean(target[0], str(target[0]) + '.o')
    return (target, source)