import os
import hug
DIRECTORY = os.path.dirname(os.path.realpath(__file__))

@hug.get('/get/document', output=hug.output_format.html)
def nagiosCommandHelp(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns command help document when no command is specified\n    '
    with open(os.path.join(DIRECTORY, 'document.html')) as document:
        return document.read()