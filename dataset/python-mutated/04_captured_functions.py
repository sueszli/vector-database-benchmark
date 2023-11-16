"""
In this example the use of captured functions is demonstrated. Like the
main function, they have access to the configuration parameters by just
accepting them as arguments.

When calling a captured function we do not need to specify the parameters that
we want to be taken from the configuration. They will automatically be filled
by Sacred. But we can always override that by passing them in explicitly.

When run, this example will output the following::

  $ ./04_captured_functions.py -l WARNING
  WARNING - captured_functions - No observers have been added to this run
  This is printed by function foo.
  This is printed by function bar.
  Overriding the default message for foo.

"""
from sacred import Experiment
ex = Experiment('captured_functions')

@ex.config
def cfg():
    if False:
        print('Hello World!')
    message = 'This is printed by function {}.'

@ex.capture
def foo(message):
    if False:
        while True:
            i = 10
    print(message.format('foo'))

@ex.capture
def bar(message):
    if False:
        print('Hello World!')
    print(message.format('bar'))

@ex.automain
def main():
    if False:
        for i in range(10):
            print('nop')
    foo()
    bar()
    foo('Overriding the default message for {}.')