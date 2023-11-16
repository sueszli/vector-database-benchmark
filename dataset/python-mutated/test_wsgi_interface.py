import re
import sys
import falcon
import falcon.testing as testing

class TestWSGIInterface:

    def test_srmock(self):
        if False:
            i = 10
            return i + 15
        mock = testing.StartResponseMock()
        mock(falcon.HTTP_200, ())
        assert mock.status == falcon.HTTP_200
        assert mock.exc_info is None
        mock = testing.StartResponseMock()
        exc_info = sys.exc_info()
        mock(falcon.HTTP_200, (), exc_info)
        assert mock.exc_info == exc_info

    def test_pep3333(self):
        if False:
            i = 10
            return i + 15
        api = falcon.App()
        mock = testing.StartResponseMock()
        response = api(testing.create_environ(), mock)
        assert _is_iterable(response)
        assert mock.call_count == 1
        assert isinstance(mock.status, str)
        assert re.match('^\\d+[a-zA-Z\\s]+$', mock.status)
        assert isinstance(mock.headers, list)
        if len(mock.headers) != 0:
            header = mock.headers[0]
            assert isinstance(header, tuple)
            assert len(header) == 2
            assert isinstance(header[0], str)
            assert isinstance(header[1], str)

def _is_iterable(thing):
    if False:
        print('Hello World!')
    try:
        for i in thing:
            break
        return True
    except TypeError:
        return False