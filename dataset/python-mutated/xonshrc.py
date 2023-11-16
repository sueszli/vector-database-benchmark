from xonsh.built_ins import XSH
env = XSH.env
env['PATH'].append('/home/scopatz/sandbox/bin')
env['LD_LIBRARY_PATH'] = ['/home/scopatz/.local/lib', '/home/scopatz/miniconda3/lib']

def _quit_awesome(args, stdin=None):
    if False:
        for i in range(10):
            print('nop')
    print('awesome python code')
XSH.aliases['qa'] = _quit_awesome
XSH.aliases['gc'] = ['git', 'commit']
env['MULTILINE_PROMPT'] = '`·.,¸,.·*¯`·.,¸,.·*¯'
env['XONSH_SHOW_TRACEBACK'] = True
env['XONSH_STORE_STDOUT'] = True
env['XONSH_HISTORY_MATCH_ANYWHERE'] = True
env['COMPLETIONS_CONFIRM'] = True
env['XONSH_AUTOPAIR'] = True