"""
This module is to hold logic which overrides default SCons behaviors to enable
ninja file generation
"""
import SCons

def ninja_hack_linkcom(env):
    if False:
        return 10
    if env['PLATFORM'] == 'win32':
        from SCons.Tool.mslink import compositeLinkAction
        if env.get('LINKCOM', None) == compositeLinkAction:
            env['LINKCOM'] = '${TEMPFILE("$LINK $LINKFLAGS /OUT:$TARGET.windows $_LIBDIRFLAGS $_LIBFLAGS $_PDB $SOURCES.windows", "$LINKCOMSTR")}'
            env['SHLINKCOM'] = '${TEMPFILE("$SHLINK $SHLINKFLAGS $_SHLINK_TARGETS $_LIBDIRFLAGS $_LIBFLAGS $_PDB $_SHLINK_SOURCES", "$SHLINKCOMSTR")}'

def ninja_hack_arcom(env):
    if False:
        print('Hello World!')
    "\n        Force ARCOM so use 's' flag on ar instead of separately running ranlib\n    "
    if env['PLATFORM'] != 'win32' and env.get('RANLIBCOM'):
        old_arflags = str(env['ARFLAGS'])
        if 's' not in old_arflags:
            old_arflags += 's'
        env['ARFLAGS'] = SCons.Util.CLVar([old_arflags])
        env['RANLIBCOM'] = ''

class NinjaNoResponseFiles(SCons.Platform.TempFileMunge):
    """Overwrite the __call__ method of SCons' TempFileMunge to not delete."""

    def __call__(self, target, source, env, for_signature):
        if False:
            return 10
        return self.cmd

    def _print_cmd_str(*_args, **_kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Disable this method'
        pass

def ninja_always_serial(self, num, taskmaster):
    if False:
        for i in range(10):
            print('nop')
    'Replacement for SCons.Job.Jobs constructor which always uses the Serial Job class.'
    self.num_jobs = num
    self.job = SCons.Taskmaster.Job.Serial(taskmaster)

class AlwaysExecAction(SCons.Action.FunctionAction):
    """Override FunctionAction.__call__ to always execute."""

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['execute'] = 1
        return super().__call__(*args, **kwargs)