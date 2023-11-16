import pytest_bdd as bdd
bdd.scenarios('completion.feature')

@bdd.then(bdd.parsers.parse('the completion model should be {model}'))
def check_model(quteproc, model):
    if False:
        i = 10
        return i + 15
    'Make sure the completion model was set to something.'
    pattern = 'Starting {} completion *'.format(model)
    quteproc.wait_for(message=pattern)