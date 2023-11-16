import json
from typing import Any, Dict
from urllib.parse import quote
import pytest
from telegram import InputFile, InputMediaPhoto, InputMediaVideo, MessageEntity
from telegram.request import RequestData
from telegram.request._requestparameter import RequestParameter
from tests.auxil.files import data_file
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def inputfiles() -> Dict[bool, InputFile]:
    if False:
        for i in range(10):
            print('nop')
    return {True: InputFile(obj='data', attach=True), False: InputFile(obj='data', attach=False)}

@pytest.fixture(scope='module')
def input_media_video() -> InputMediaVideo:
    if False:
        i = 10
        return i + 15
    return InputMediaVideo(media=data_file('telegram.mp4').read_bytes(), thumbnail=data_file('telegram.jpg').read_bytes(), parse_mode=None)

@pytest.fixture(scope='module')
def input_media_photo() -> InputMediaPhoto:
    if False:
        while True:
            i = 10
    return InputMediaPhoto(media=data_file('telegram.jpg').read_bytes(), parse_mode=None)

@pytest.fixture(scope='module')
def simple_params() -> Dict[str, Any]:
    if False:
        return 10
    return {'string': 'string', 'integer': 1, 'tg_object': MessageEntity('type', 1, 1), 'list': [1, 'string', MessageEntity('type', 1, 1)]}

@pytest.fixture(scope='module')
def simple_jsons() -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    return {'string': 'string', 'integer': json.dumps(1), 'tg_object': MessageEntity('type', 1, 1).to_json(), 'list': json.dumps([1, 'string', MessageEntity('type', 1, 1).to_dict()])}

@pytest.fixture(scope='module')
def simple_rqs(simple_params) -> RequestData:
    if False:
        while True:
            i = 10
    return RequestData([RequestParameter.from_input(key, value) for (key, value) in simple_params.items()])

@pytest.fixture(scope='module')
def file_params(inputfiles, input_media_video, input_media_photo) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    return {'inputfile_attach': inputfiles[True], 'inputfile_no_attach': inputfiles[False], 'inputmedia': input_media_video, 'inputmedia_list': [input_media_video, input_media_photo]}

@pytest.fixture(scope='module')
def file_jsons(inputfiles, input_media_video, input_media_photo) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    input_media_video_dict = input_media_video.to_dict()
    input_media_video_dict['media'] = input_media_video.media.attach_uri
    input_media_video_dict['thumbnail'] = input_media_video.thumbnail.attach_uri
    input_media_photo_dict = input_media_photo.to_dict()
    input_media_photo_dict['media'] = input_media_photo.media.attach_uri
    return {'inputfile_attach': inputfiles[True].attach_uri, 'inputmedia': json.dumps(input_media_video_dict), 'inputmedia_list': json.dumps([input_media_video_dict, input_media_photo_dict])}

@pytest.fixture(scope='module')
def file_rqs(file_params) -> RequestData:
    if False:
        while True:
            i = 10
    return RequestData([RequestParameter.from_input(key, value) for (key, value) in file_params.items()])

@pytest.fixture(scope='module')
def mixed_params(file_params, simple_params) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    both = file_params.copy()
    both.update(simple_params)
    return both

@pytest.fixture(scope='module')
def mixed_jsons(file_jsons, simple_jsons) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    both = file_jsons.copy()
    both.update(simple_jsons)
    return both

@pytest.fixture(scope='module')
def mixed_rqs(mixed_params) -> RequestData:
    if False:
        return 10
    return RequestData([RequestParameter.from_input(key, value) for (key, value) in mixed_params.items()])

class TestRequestDataWithoutRequest:

    def test_slot_behaviour(self, simple_rqs):
        if False:
            return 10
        for attr in simple_rqs.__slots__:
            assert getattr(simple_rqs, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(simple_rqs)) == len(set(mro_slots(simple_rqs))), 'duplicate slot'

    def test_contains_files(self, simple_rqs, file_rqs, mixed_rqs):
        if False:
            print('Hello World!')
        assert not simple_rqs.contains_files
        assert file_rqs.contains_files
        assert mixed_rqs.contains_files

    def test_parameters(self, simple_rqs, file_rqs, mixed_rqs, inputfiles, input_media_video, input_media_photo):
        if False:
            i = 10
            return i + 15
        simple_params_expected = {'string': 'string', 'integer': 1, 'tg_object': MessageEntity('type', 1, 1).to_dict(), 'list': [1, 'string', MessageEntity('type', 1, 1).to_dict()]}
        video_value = {'media': input_media_video.media.attach_uri, 'thumbnail': input_media_video.thumbnail.attach_uri, 'type': input_media_video.type}
        photo_value = {'media': input_media_photo.media.attach_uri, 'type': input_media_photo.type}
        file_params_expected = {'inputfile_attach': inputfiles[True].attach_uri, 'inputmedia': video_value, 'inputmedia_list': [video_value, photo_value]}
        mixed_params_expected = simple_params_expected.copy()
        mixed_params_expected.update(file_params_expected)
        assert simple_rqs.parameters == simple_params_expected
        assert file_rqs.parameters == file_params_expected
        assert mixed_rqs.parameters == mixed_params_expected

    def test_json_parameters(self, simple_rqs, file_rqs, mixed_rqs, simple_jsons, file_jsons, mixed_jsons):
        if False:
            return 10
        assert simple_rqs.json_parameters == simple_jsons
        assert file_rqs.json_parameters == file_jsons
        assert mixed_rqs.json_parameters == mixed_jsons

    def test_json_payload(self, simple_rqs, file_rqs, mixed_rqs, simple_jsons, file_jsons, mixed_jsons):
        if False:
            return 10
        assert simple_rqs.json_payload == json.dumps(simple_jsons).encode()
        assert file_rqs.json_payload == json.dumps(file_jsons).encode()
        assert mixed_rqs.json_payload == json.dumps(mixed_jsons).encode()

    def test_multipart_data(self, simple_rqs, file_rqs, mixed_rqs, inputfiles, input_media_video, input_media_photo):
        if False:
            while True:
                i = 10
        expected = {inputfiles[True].attach_name: inputfiles[True].field_tuple, 'inputfile_no_attach': inputfiles[False].field_tuple, input_media_photo.media.attach_name: input_media_photo.media.field_tuple, input_media_video.media.attach_name: input_media_video.media.field_tuple, input_media_video.thumbnail.attach_name: input_media_video.thumbnail.field_tuple}
        assert simple_rqs.multipart_data == {}
        assert file_rqs.multipart_data == expected
        assert mixed_rqs.multipart_data == expected

    def test_url_encoding(self):
        if False:
            i = 10
            return i + 15
        data = RequestData([RequestParameter.from_input('chat_id', 123), RequestParameter.from_input('text', 'Hello there/!')])
        expected_params = 'chat_id=123&text=Hello+there%2F%21'
        expected_url = 'https://te.st/method?' + expected_params
        assert data.url_encoded_parameters() == expected_params
        assert data.parametrized_url('https://te.st/method') == expected_url
        expected_params = 'chat_id=123&text=Hello%20there/!'
        expected_url = 'https://te.st/method?' + expected_params
        assert data.url_encoded_parameters(encode_kwargs={'quote_via': quote, 'safe': '/!'}) == expected_params
        assert data.parametrized_url('https://te.st/method', encode_kwargs={'quote_via': quote, 'safe': '/!'}) == expected_url