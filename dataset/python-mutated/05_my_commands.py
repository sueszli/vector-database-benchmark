"""
This experiment showcases the concept of commands in Sacred.
By just using the ``@ex.command`` decorator we can add additional commands to
the command-line interface of the experiment::

  $ ./05_my_commands.py greet
  WARNING - my_commands - No observers have been added to this run
  INFO - my_commands - Running command 'greet'
  INFO - my_commands - Started
  Hello John! Nice to greet you!
  INFO - my_commands - Completed after 0:00:00

::

  $ ./05_my_commands.py shout
  WARNING - my_commands - No observers have been added to this run
  INFO - my_commands - Running command 'shout'
  INFO - my_commands - Started
  WHAZZZUUUUUUUUUUP!!!????
  INFO - my_commands - Completed after 0:00:00

Of course we can also use ``with`` and other flags with those commands::

  $ ./05_my_commands.py greet with name='Jane' -l WARNING
  WARNING - my_commands - No observers have been added to this run
  Hello Jane! Nice to greet you!

In fact, the main function is also just a command::

  $ ./05_my_commands.py main
  WARNING - my_commands - No observers have been added to this run
  INFO - my_commands - Running command 'main'
  INFO - my_commands - Started
  This is just the main command. Try greet or shout.
  INFO - my_commands - Completed after 0:00:00

Commands also appear in the help text, and you can get additional information
about all commands using ``./05_my_commands.py help [command]``.
"""
from sacred import Experiment
ex = Experiment('my_commands')

@ex.config
def cfg():
    if False:
        while True:
            i = 10
    name = 'John'

@ex.command
def greet(name):
    if False:
        print('Hello World!')
    '\n    Print a nice greet message.\n\n    Uses the name from config.\n    '
    print('Hello {}! Nice to greet you!'.format(name))

@ex.command
def shout():
    if False:
        while True:
            i = 10
    '\n    Shout slang question for "what is up?"\n    '
    print('WHAZZZUUUUUUUUUUP!!!????')

@ex.automain
def main():
    if False:
        print('Hello World!')
    print('This is just the main command. Try greet or shout.')