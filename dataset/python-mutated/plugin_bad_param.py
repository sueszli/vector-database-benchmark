from graphite.functions.params import Param, ParamTypes

def test(seriesList):
    if False:
        i = 10
        return i + 15
    'This is a test function'
    return seriesList
test.group = 'Test'
test.params = [Param('seriesList', ParamTypes.seriesList, required=True), 'bad param']
SeriesFunctions = {'testFunc': test}

def pieTest(series):
    if False:
        while True:
            i = 10
    'This is a test pie function'
    return max(series)
pieTest.group = 'Test'
pieTest.params = [Param('series', ParamTypes.series, required=True), 'bad param']
PieFunctions = {'testFunc': pieTest}