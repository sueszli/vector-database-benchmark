from docker.utils.json_stream import json_splitter, stream_as_text, json_stream

class TestJsonSplitter:

    def test_json_splitter_no_object(self):
        if False:
            i = 10
            return i + 15
        data = '{"foo": "bar'
        assert json_splitter(data) is None

    def test_json_splitter_with_object(self):
        if False:
            i = 10
            return i + 15
        data = '{"foo": "bar"}\n  \n{"next": "obj"}'
        assert json_splitter(data) == ({'foo': 'bar'}, '{"next": "obj"}')

    def test_json_splitter_leading_whitespace(self):
        if False:
            print('Hello World!')
        data = '\n   \r{"foo": "bar"}\n\n   {"next": "obj"}'
        assert json_splitter(data) == ({'foo': 'bar'}, '{"next": "obj"}')

class TestStreamAsText:

    def test_stream_with_non_utf_unicode_character(self):
        if False:
            return 10
        stream = [b'\xed\xf3\xf3']
        (output,) = stream_as_text(stream)
        assert output == '���'

    def test_stream_with_utf_character(self):
        if False:
            while True:
                i = 10
        stream = ['ěĝ'.encode()]
        (output,) = stream_as_text(stream)
        assert output == 'ěĝ'

class TestJsonStream:

    def test_with_falsy_entries(self):
        if False:
            print('Hello World!')
        stream = ['{"one": "two"}\n{}\n', '[1, 2, 3]\n[]\n']
        output = list(json_stream(stream))
        assert output == [{'one': 'two'}, {}, [1, 2, 3], []]

    def test_with_leading_whitespace(self):
        if False:
            print('Hello World!')
        stream = ['\n  \r\n  {"one": "two"}{"x": 1}', '  {"three": "four"}\t\t{"x": 2}']
        output = list(json_stream(stream))
        assert output == [{'one': 'two'}, {'x': 1}, {'three': 'four'}, {'x': 2}]