import vim
from vimspector import settings, utils

def SignDefined(name):
    if False:
        return 10
    if utils.Exists('*sign_getdefined'):
        return int(vim.eval(f"len( sign_getdefined( '{utils.Escape(name)}' ) )"))
    return False

def DefineSign(name, text, double_text, texthl, col='right', **kwargs):
    if False:
        print('Hello World!')
    if utils.GetVimValue(vim.options, 'ambiwidth', '') == 'double':
        text = double_text
    if text is not None:
        if col == 'right':
            if int(utils.Call('strdisplaywidth', text)) < 2:
                text = ' ' + text
        text = text.replace(' ', '\\ ')
        kwargs['text'] = text
    if texthl is not None:
        kwargs['texthl'] = texthl
    cmd = f'sign define {name}'
    for (key, value) in kwargs.items():
        cmd += f' {key}={value}'
    vim.command(cmd)

def PlaceSign(sign_id, group, name, file_name, line):
    if False:
        for i in range(10):
            print('nop')
    priority = settings.Dict('sign_priority')[name]
    cmd = f'sign place {sign_id} group={group} name={name} priority={priority} line={line} file={file_name}'
    vim.command(cmd)

def UnplaceSign(sign_id, group):
    if False:
        while True:
            i = 10
    vim.command(f'sign unplace {sign_id} group={group}')

def DefineProgramCounterSigns():
    if False:
        while True:
            i = 10
    if not SignDefined('vimspectorPC'):
        DefineSign('vimspectorPC', text='▶', double_text='▶', texthl='MatchParen', linehl='CursorLine')
    if not SignDefined('vimspectorPCBP'):
        DefineSign('vimspectorPCBP', text='●▶', double_text='▷', texthl='MatchParen', linehl='CursorLine')
    if not SignDefined('vimspectorNonActivePC'):
        DefineSign('vimspectorNonActivePC', text=None, double_text=None, texthl=None, linehl='DiffAdd')