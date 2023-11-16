from graphite.functions.params import Param, ParamTypes

def test(seriesList):
    if False:
        i = 10
        return i + 15
    'This is a test function'
    return seriesList
test.group = 'Test'
test.params = [Param('seriesList', ParamTypes.bad, required=True)]
SeriesFunctions = {'testFunc': test}