"""An example of how to embed an IPython shell into a running program.

Please see the documentation in the IPython.Shell module for more details.

The accompanying file embed_class_short.py has quick code fragments for
embedding which you can cut and paste in your code once you understand how
things work.

The code in this file is deliberately extra-verbose, meant for learning."""
from IPython.terminal.prompts import Prompts, Token

class CustomPrompt(Prompts):

    def in_prompt_tokens(self, cli=None):
        if False:
            print('Hello World!')
        return [(Token.Prompt, 'In <'), (Token.PromptNum, str(self.shell.execution_count)), (Token.Prompt, '>: ')]

    def out_prompt_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        return [(Token.OutPrompt, 'Out<'), (Token.OutPromptNum, str(self.shell.execution_count)), (Token.OutPrompt, '>: ')]
from traitlets.config.loader import Config
try:
    get_ipython
except NameError:
    nested = 0
    cfg = Config()
    cfg.TerminalInteractiveShell.prompts_class = CustomPrompt
else:
    print('Running nested copies of IPython.')
    print('The prompts for the nested copy have been modified')
    cfg = Config()
    nested = 1
from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed(config=cfg, banner1='Dropping into IPython', exit_msg='Leaving Interpreter, back to program.')
ipshell2 = InteractiveShellEmbed(config=cfg, banner1='Second IPython instance.')
print('\nHello. This is printed from the main controller program.\n')
ipshell('***Called from top level. Hit Ctrl-D to exit interpreter and continue program.\nNote that if you use %kill_embedded, you can fully deactivate\nThis embedded instance so it will never turn on again')
print('\nBack in caller program, moving along...\n')
ipshell.banner2 = 'Entering interpreter - New Banner'
ipshell.exit_msg = 'Leaving interpreter - New exit_msg'

def foo(m):
    if False:
        while True:
            i = 10
    s = 'spam'
    ipshell('***In foo(). Try %whos, or print s or m:')
    print('foo says m = ', m)

def bar(n):
    if False:
        print('Hello World!')
    s = 'eggs'
    ipshell('***In bar(). Try %whos, or print s or n:')
    print('bar says n = ', n)
print('Main program calling foo("eggs")\n')
foo('eggs')
ipshell.dummy_mode = True
print('\nTrying to call IPython which is now "dummy":')
ipshell()
print('Nothing happened...')
print('\nOverriding dummy mode manually:')
ipshell(dummy=False)
ipshell.dummy_mode = False
print('You can even have multiple embedded instances:')
ipshell2()
print('\nMain program calling bar("spam")\n')
bar('spam')
print('Main program finished. Bye!')