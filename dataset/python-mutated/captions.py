import math
import os
import time
import json
import xml.etree.ElementTree as ElementTree
from html import unescape
from typing import Dict, Optional
from pytube import request
from pytube.helpers import safe_filename, target_directory

class Caption:
    """Container for caption tracks."""

    def __init__(self, caption_track: Dict):
        if False:
            return 10
        'Construct a :class:`Caption <Caption>`.\n\n        :param dict caption_track:\n            Caption track data extracted from ``watch_html``.\n        '
        self.url = caption_track.get('baseUrl')
        name_dict = caption_track['name']
        if 'simpleText' in name_dict:
            self.name = name_dict['simpleText']
        else:
            for el in name_dict['runs']:
                if 'text' in el:
                    self.name = el['text']
        self.code = caption_track['vssId']
        self.code = self.code.strip('.')

    @property
    def xml_captions(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Download the xml caption tracks.'
        return request.get(self.url)

    @property
    def json_captions(self) -> dict:
        if False:
            return 10
        'Download and parse the json caption tracks.'
        json_captions_url = self.url.replace('fmt=srv3', 'fmt=json3')
        text = request.get(json_captions_url)
        parsed = json.loads(text)
        assert parsed['wireMagic'] == 'pb3', 'Unexpected captions format'
        return parsed

    def generate_srt_captions(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Generate "SubRip Subtitle" captions.\n\n        Takes the xml captions from :meth:`~pytube.Caption.xml_captions` and\n        recompiles them into the "SubRip Subtitle" format.\n        '
        return self.xml_caption_to_srt(self.xml_captions)

    @staticmethod
    def float_to_srt_time_format(d: float) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Convert decimal durations into proper srt format.\n\n        :rtype: str\n        :returns:\n            SubRip Subtitle (str) formatted time duration.\n\n        float_to_srt_time_format(3.89) -> '00:00:03,890'\n        "
        (fraction, whole) = math.modf(d)
        time_fmt = time.strftime('%H:%M:%S,', time.gmtime(whole))
        ms = f'{fraction:.3f}'.replace('0.', '')
        return time_fmt + ms

    def xml_caption_to_srt(self, xml_captions: str) -> str:
        if False:
            i = 10
            return i + 15
        'Convert xml caption tracks to "SubRip Subtitle (srt)".\n\n        :param str xml_captions:\n            XML formatted caption tracks.\n        '
        segments = []
        root = ElementTree.fromstring(xml_captions)
        for (i, child) in enumerate(list(root)):
            text = child.text or ''
            caption = unescape(text.replace('\n', ' ').replace('  ', ' '))
            try:
                duration = float(child.attrib['dur'])
            except KeyError:
                duration = 0.0
            start = float(child.attrib['start'])
            end = start + duration
            sequence_number = i + 1
            line = '{seq}\n{start} --> {end}\n{text}\n'.format(seq=sequence_number, start=self.float_to_srt_time_format(start), end=self.float_to_srt_time_format(end), text=caption)
            segments.append(line)
        return '\n'.join(segments).strip()

    def download(self, title: str, srt: bool=True, output_path: Optional[str]=None, filename_prefix: Optional[str]=None) -> str:
        if False:
            i = 10
            return i + 15
        'Write the media stream to disk.\n\n        :param title:\n            Output filename (stem only) for writing media file.\n            If one is not specified, the default filename is used.\n        :type title: str\n        :param srt:\n            Set to True to download srt, false to download xml. Defaults to True.\n        :type srt bool\n        :param output_path:\n            (optional) Output path for writing media file. If one is not\n            specified, defaults to the current working directory.\n        :type output_path: str or None\n        :param filename_prefix:\n            (optional) A string that will be prepended to the filename.\n            For example a number in a playlist or the name of a series.\n            If one is not specified, nothing will be prepended\n            This is separate from filename so you can use the default\n            filename but still add a prefix.\n        :type filename_prefix: str or None\n\n        :rtype: str\n        '
        if title.endswith('.srt') or title.endswith('.xml'):
            filename = '.'.join(title.split('.')[:-1])
        else:
            filename = title
        if filename_prefix:
            filename = f'{safe_filename(filename_prefix)}{filename}'
        filename = safe_filename(filename)
        filename += f' ({self.code})'
        if srt:
            filename += '.srt'
        else:
            filename += '.xml'
        file_path = os.path.join(target_directory(output_path), filename)
        with open(file_path, 'w', encoding='utf-8') as file_handle:
            if srt:
                file_handle.write(self.generate_srt_captions())
            else:
                file_handle.write(self.xml_captions)
        return file_path

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Printable object representation.'
        return '<Caption lang="{s.name}" code="{s.code}">'.format(s=self)