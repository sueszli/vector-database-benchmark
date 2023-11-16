""" A very configurable Hello World. Yay! """
from sacred import Experiment
ex = Experiment('hello_config')

@ex.named_config
def rude():
    if False:
        return 10
    'A rude named config'
    recipient = 'bastard'
    message = 'Fuck off you {}!'.format(recipient)

@ex.config
def cfg():
    if False:
        return 10
    recipient = 'world'
    message = 'Hello {}!'.format(recipient)

@ex.automain
def main(message):
    if False:
        print('Hello World!')
    print(__name__)
    print(message)