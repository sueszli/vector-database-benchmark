import pytest_bdd as bdd
bdd.scenarios('zoom.feature')

@bdd.then(bdd.parsers.parse('the zoom should be {zoom}%'))
def check_zoom(quteproc, zoom):
    if False:
        while True:
            i = 10
    data = quteproc.get_session()
    histories = data['windows'][0]['tabs'][0]['history']
    value = next((h for h in histories if 'zoom' in h))['zoom'] * 100
    assert abs(value - float(zoom)) < 0.0001