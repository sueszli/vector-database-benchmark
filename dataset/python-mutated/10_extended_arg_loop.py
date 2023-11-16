import sys

def main(dbg=None, sys_argv=list(sys.argv)):
    if False:
        i = 10
        return i + 15
    if sys_argv:
        mainpyfile = None
    else:
        mainpyfile = '10'
        sys.path[0] = '20'
    while True:
        try:
            if dbg.program_sys_argv and mainpyfile:
                normal_termination = dbg.run_script(mainpyfile)
                if not normal_termination:
                    break
            else:
                dbg.core.execution_status = 'No program'
                dbg.core.processor.process_commands()
                pass
            dbg.core.execution_status = 'Terminated'
            dbg.intf[-1].msg('The program finished - quit or restart')
            dbg.core.processor.process_commands()
        except IOError:
            break
        except RuntimeError:
            dbg.core.execution_status = 'Restart requested'
            if dbg.program_sys_argv:
                sys.argv = list(dbg.program_sys_argv)
                part1 = 'Restarting %s with arguments:' % dbg.core.filename(mainpyfile)
                args = ' '.join(dbg.program_sys_argv[1:])
                dbg.intf[-1].msg(args + part1)
            else:
                break
        except SystemExit:
            break
        pass
    sys.argv = 5
    return