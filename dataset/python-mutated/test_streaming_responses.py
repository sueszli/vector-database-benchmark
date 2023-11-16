import pytest
import os.path as path

class TestFlaskStreaming(object):

    @pytest.mark.ckan_config('ckan.plugins', 'example_flask_streaming')
    def test_accordance_of_chunks(self, app):
        if False:
            while True:
                i = 10
        'Test streaming of items collection.'
        resp = app.get('/stream/string')
        expected = b'Hello World, this is served from an extension'
        assert expected == b' '.join(resp.iter_encoded())

    @pytest.mark.ckan_config('ckan.plugins', 'example_flask_streaming')
    def test_template_streaming(self, app):
        if False:
            return 10
        'Test streaming of template response.'
        bound = 7
        url = '/stream/template/{}'.format(bound)
        resp = app.get(url)
        content = b''.join(resp.iter_encoded())
        for i in range(bound):
            assert bytes(str(i).encode()) in content

    @pytest.mark.ckan_config('ckan.plugins', 'example_flask_streaming')
    def test_file_streaming(self, app):
        if False:
            return 10
        'Test streaming of existing file(10lines.txt).'
        resp = app.get('/stream/file')
        f_path = path.join(path.dirname(path.abspath(__file__)), '10lines.txt')
        with open(f_path) as test_file:
            content = [bytes(line.encode()) for line in test_file.readlines()]
            assert content == list(resp.iter_encoded())

    @pytest.mark.ckan_config('ckan.plugins', 'example_flask_streaming')
    def test_render_with_context(self, app):
        if False:
            return 10
        'Test availability of context inside templates.'
        resp = app.get('/stream/context?var=10')
        assert b'10' == resp.data