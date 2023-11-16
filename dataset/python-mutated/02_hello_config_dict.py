""" A configurable Hello World "experiment".
In this example we configure the message using a dictionary with
``ex.add_config``

You can run it like this::

  $ ./02_hello_config_dict.py
  WARNING - 02_hello_config_dict - No observers have been added to this run
  INFO - 02_hello_config_dict - Running command 'main'
  INFO - 02_hello_config_dict - Started
  Hello world!
  INFO - 02_hello_config_dict - Completed after 0:00:00

The message can also easily be changed using the ``with`` command-line
argument::

  $ ./02_hello_config_dict.py with message='Ciao world!'
  WARNING - 02_hello_config_dict - No observers have been added to this run
  INFO - 02_hello_config_dict - Running command 'main'
  INFO - 02_hello_config_dict - Started
  Ciao world!
  INFO - 02_hello_config_dict - Completed after 0:00:00
"""
from sacred import Experiment
ex = Experiment()
ex.add_config({'message': 'Hello world!'})

@ex.automain
def main(message):
    if False:
        print('Hello World!')
    print(message)