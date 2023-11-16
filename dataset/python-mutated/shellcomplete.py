from __future__ import absolute_import
import sys

def shellcomplete(context=None, outfile=None):
    if False:
        for i in range(10):
            print('nop')
    if outfile is None:
        outfile = sys.stdout
    if context is None:
        shellcomplete_commands(outfile=outfile)
    else:
        shellcomplete_on_command(context, outfile=outfile)

def shellcomplete_on_command(cmdname, outfile=None):
    if False:
        for i in range(10):
            print('nop')
    cmdname = str(cmdname)
    if outfile is None:
        outfile = sys.stdout
    from inspect import getdoc
    import commands
    cmdobj = commands.get_cmd_object(cmdname)
    doc = getdoc(cmdobj)
    if doc is None:
        raise NotImplementedError('sorry, no detailed shellcomplete yet for %r' % cmdname)
    shellcomplete_on_options(cmdobj.options().values(), outfile=outfile)
    for aname in cmdobj.takes_args:
        outfile.write(aname + '\n')

def shellcomplete_on_options(options, outfile=None):
    if False:
        while True:
            i = 10
    for opt in options:
        short_name = opt.short_name()
        if short_name:
            outfile.write('"(--%s -%s)"{--%s,-%s}\n' % (opt.name, short_name, opt.name, short_name))
        else:
            outfile.write('--%s\n' % opt.name)

def shellcomplete_commands(outfile=None):
    if False:
        return 10
    'List all commands'
    from bzrlib import commands
    from inspect import getdoc
    commands.install_bzr_command_hooks()
    if outfile is None:
        outfile = sys.stdout
    cmds = []
    for cmdname in commands.all_command_names():
        cmd = commands.get_cmd_object(cmdname)
        cmds.append((cmdname, cmd))
        for alias in cmd.aliases:
            cmds.append((alias, cmd))
    cmds.sort()
    for (cmdname, cmd) in cmds:
        if cmd.hidden:
            continue
        doc = getdoc(cmd)
        if doc is None:
            outfile.write(cmdname + '\n')
        else:
            doclines = doc.splitlines()
            firstline = doclines[0].lower()
            outfile.write(cmdname + ':' + firstline[0:-1] + '\n')