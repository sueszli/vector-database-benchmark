import os
import pytest
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch
from pytube import Caption, CaptionQuery, captions

def test_float_to_srt_time_format():
    if False:
        return 10
    caption1 = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
    assert caption1.float_to_srt_time_format(3.89) == '00:00:03,890'

def test_caption_query_sequence():
    if False:
        i = 10
        return i + 15
    caption1 = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
    caption2 = Caption({'url': 'url2', 'name': {'simpleText': 'name2'}, 'languageCode': 'fr', 'vssId': '.fr'})
    caption_query = CaptionQuery(captions=[caption1, caption2])
    assert len(caption_query) == 2
    assert caption_query['en'] == caption1
    assert caption_query['fr'] == caption2
    with pytest.raises(KeyError):
        assert caption_query['nada'] is not None

def test_caption_query_get_by_language_code_when_exists():
    if False:
        i = 10
        return i + 15
    caption1 = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
    caption2 = Caption({'url': 'url2', 'name': {'simpleText': 'name2'}, 'languageCode': 'fr', 'vssId': '.fr'})
    caption_query = CaptionQuery(captions=[caption1, caption2])
    assert caption_query['en'] == caption1

def test_caption_query_get_by_language_code_when_not_exists():
    if False:
        return 10
    caption1 = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
    caption2 = Caption({'url': 'url2', 'name': {'simpleText': 'name2'}, 'languageCode': 'fr', 'vssId': '.fr'})
    caption_query = CaptionQuery(captions=[caption1, caption2])
    with pytest.raises(KeyError):
        assert caption_query['hello'] is not None

@mock.patch('pytube.captions.Caption.generate_srt_captions')
def test_download(srt):
    if False:
        return 10
    open_mock = mock_open()
    with patch('builtins.open', open_mock):
        srt.return_value = ''
        caption = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
        caption.download('title')
        assert open_mock.call_args_list[0][0][0].split(os.path.sep)[-1] == 'title (en).srt'

@mock.patch('pytube.captions.Caption.generate_srt_captions')
def test_download_with_prefix(srt):
    if False:
        for i in range(10):
            print('nop')
    open_mock = mock_open()
    with patch('builtins.open', open_mock):
        srt.return_value = ''
        caption = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
        caption.download('title', filename_prefix='1 ')
        assert open_mock.call_args_list[0][0][0].split(os.path.sep)[-1] == '1 title (en).srt'

@mock.patch('pytube.captions.Caption.generate_srt_captions')
def test_download_with_output_path(srt):
    if False:
        return 10
    open_mock = mock_open()
    captions.target_directory = MagicMock(return_value='/target')
    with patch('builtins.open', open_mock):
        srt.return_value = ''
        caption = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
        file_path = caption.download('title', output_path='blah')
        assert file_path == os.path.join('/target', 'title (en).srt')
        captions.target_directory.assert_called_with('blah')

@mock.patch('pytube.captions.Caption.xml_captions')
def test_download_xml_and_trim_extension(xml):
    if False:
        i = 10
        return i + 15
    open_mock = mock_open()
    with patch('builtins.open', open_mock):
        xml.return_value = ''
        caption = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
        caption.download('title.xml', srt=False)
        assert open_mock.call_args_list[0][0][0].split(os.path.sep)[-1] == 'title (en).xml'

def test_repr():
    if False:
        print('Hello World!')
    caption = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
    assert str(caption) == '<Caption lang="name1" code="en">'
    caption_query = CaptionQuery(captions=[caption])
    assert repr(caption_query) == '{\'en\': <Caption lang="name1" code="en">}'

@mock.patch('pytube.request.get')
def test_xml_captions(request_get):
    if False:
        return 10
    request_get.return_value = 'test'
    caption = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
    assert caption.xml_captions == 'test'

@mock.patch('pytube.captions.request')
def test_generate_srt_captions(request):
    if False:
        return 10
    request.get.return_value = '<?xml version="1.0" encoding="utf-8" ?><transcript><text start="6.5" dur="1.7">[Herb, Software Engineer]\n本影片包含隱藏式字幕。</text><text start="8.3" dur="2.7">如要啓動字幕，請按一下這裡的圖示。</text></transcript>'
    caption = Caption({'url': 'url1', 'name': {'simpleText': 'name1'}, 'languageCode': 'en', 'vssId': '.en'})
    assert caption.generate_srt_captions() == '1\n00:00:06,500 --> 00:00:08,200\n[Herb, Software Engineer] 本影片包含隱藏式字幕。\n\n2\n00:00:08,300 --> 00:00:11,000\n如要啓動字幕，請按一下這裡的圖示。'