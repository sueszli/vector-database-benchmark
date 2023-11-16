raise RuntimeError('Bad app!')

def app(environ, start_response):
    if False:
        return 10
    assert 1 == 2, "Shouldn't get here."