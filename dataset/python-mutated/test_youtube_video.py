import hashlib
from unittest.mock import MagicMock, Mock, patch
import pytest
from embedchain.loaders.youtube_video import YoutubeVideoLoader

@pytest.fixture
def youtube_video_loader():
    if False:
        i = 10
        return i + 15
    return YoutubeVideoLoader()

def test_load_data(youtube_video_loader):
    if False:
        i = 10
        return i + 15
    video_url = 'https://www.youtube.com/watch?v=VIDEO_ID'
    mock_loader = Mock()
    mock_page_content = 'This is a YouTube video content.'
    mock_loader.load.return_value = [MagicMock(page_content=mock_page_content, metadata={'url': video_url, 'title': 'Test Video'})]
    with patch('embedchain.loaders.youtube_video.YoutubeLoader.from_youtube_url', return_value=mock_loader):
        result = youtube_video_loader.load_data(video_url)
    expected_doc_id = hashlib.sha256((mock_page_content + video_url).encode()).hexdigest()
    assert result['doc_id'] == expected_doc_id
    expected_data = [{'content': 'This is a YouTube video content.', 'meta_data': {'url': video_url, 'title': 'Test Video'}}]
    assert result['data'] == expected_data

def test_load_data_with_empty_doc(youtube_video_loader):
    if False:
        i = 10
        return i + 15
    video_url = 'https://www.youtube.com/watch?v=VIDEO_ID'
    mock_loader = Mock()
    mock_loader.load.return_value = []
    with patch('embedchain.loaders.youtube_video.YoutubeLoader.from_youtube_url', return_value=mock_loader):
        with pytest.raises(ValueError):
            youtube_video_loader.load_data(video_url)