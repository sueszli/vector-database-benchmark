"""media.py unit tests that are common to st.audio + st.video"""
from enum import Enum
from unittest import mock
from parameterized import parameterized
import streamlit as st
from streamlit.cursor import make_delta_path
from streamlit.elements.media import MediaData
from streamlit.proto.RootContainer_pb2 import RootContainer
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class MockMediaKind(Enum):
    AUDIO = 'audio'
    VIDEO = 'video'

class MediaTest(DeltaGeneratorTestCase):

    @parameterized.expand([('foo.wav', 'audio/wav', MockMediaKind.AUDIO, False), ('path/to/foo.wav', 'audio/wav', MockMediaKind.AUDIO, False), (b'fake_audio_data', 'audio/wav', MockMediaKind.AUDIO, False), ('https://foo.com/foo.wav', 'audio/wav', MockMediaKind.AUDIO, True), ('foo.mp4', 'video/mp4', MockMediaKind.VIDEO, False), ('path/to/foo.mp4', 'video/mp4', MockMediaKind.VIDEO, False), (b'fake_video_data', 'video/mp4', MockMediaKind.VIDEO, False), ('https://foo.com/foo.mp4', 'video/mp4', MockMediaKind.VIDEO, True)])
    def test_add_bytes_and_filenames_to_mediafilemanager(self, media_data: MediaData, mimetype: str, media_kind: MockMediaKind, is_url: bool):
        if False:
            print('Hello World!')
        'st.audio + st.video should register bytes and filenames with the\n        MediaFileManager. URL-based media does not go through the MediaFileManager.\n        '
        with mock.patch('streamlit.runtime.media_file_manager.MediaFileManager.add') as mock_mfm_add, mock.patch('streamlit.runtime.caching.save_media_data'):
            mock_mfm_add.return_value = 'https://mockoutputurl.com'
            if media_kind is MockMediaKind.AUDIO:
                st.audio(media_data, mimetype)
                element = self.get_delta_from_queue().new_element
                element_url = element.audio.url
            else:
                st.video(media_data, mimetype)
                element = self.get_delta_from_queue().new_element
                element_url = element.video.url
            if is_url:
                self.assertEqual(media_data, element_url)
                mock_mfm_add.assert_not_called()
            else:
                mock_mfm_add.assert_called_once_with(media_data, mimetype, str(make_delta_path(RootContainer.MAIN, (), 0)))
                self.assertEqual('https://mockoutputurl.com', element_url)