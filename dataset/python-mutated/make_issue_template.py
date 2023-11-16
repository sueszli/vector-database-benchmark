import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from devscripts.utils import get_filename_args, read_file, write_file
VERBOSE_TMPL = '\n  - type: checkboxes\n    id: verbose\n    attributes:\n      label: Provide verbose output that clearly demonstrates the problem\n      options:\n        - label: Run **your** yt-dlp command with **-vU** flag added (`yt-dlp -vU <your command line>`)\n          required: true\n        - label: "If using API, add `\'verbose\': True` to `YoutubeDL` params instead"\n          required: false\n        - label: Copy the WHOLE output (starting with `[debug] Command-line config`) and insert it below\n          required: true\n  - type: textarea\n    id: log\n    attributes:\n      label: Complete Verbose Output\n      description: |\n        It should start like this:\n      placeholder: |\n        [debug] Command-line config: [\'-vU\', \'https://www.youtube.com/watch?v=BaW_jenozKc\']\n        [debug] Encodings: locale cp65001, fs utf-8, pref cp65001, out utf-8, error utf-8, screen utf-8\n        [debug] yt-dlp version nightly@... from yt-dlp/yt-dlp [b634ba742] (win_exe)\n        [debug] Python 3.8.10 (CPython 64bit) - Windows-10-10.0.22000-SP0\n        [debug] exe versions: ffmpeg N-106550-g072101bd52-20220410 (fdk,setts), ffprobe N-106624-g391ce570c8-20220415, phantomjs 2.1.1\n        [debug] Optional libraries: Cryptodome-3.15.0, brotli-1.0.9, certifi-2022.06.15, mutagen-1.45.1, sqlite3-2.6.0, websockets-10.3\n        [debug] Proxy map: {}\n        [debug] Request Handlers: urllib, requests\n        [debug] Loaded 1893 extractors\n        [debug] Fetching release info: https://api.github.com/repos/yt-dlp/yt-dlp-nightly-builds/releases/latest\n        yt-dlp is up to date (nightly@... from yt-dlp/yt-dlp-nightly-builds)\n        [youtube] Extracting URL: https://www.youtube.com/watch?v=BaW_jenozKc\n        <more lines>\n      render: shell\n    validations:\n      required: true\n'.strip()
NO_SKIP = '\n  - type: checkboxes\n    attributes:\n      label: DO NOT REMOVE OR SKIP THE ISSUE TEMPLATE\n      description: Fill all fields even if you think it is irrelevant for the issue\n      options:\n        - label: I understand that I will be **blocked** if I *intentionally* remove or skip any mandatory\\* field\n          required: true\n'.strip()

def main():
    if False:
        print('Hello World!')
    fields = {'no_skip': NO_SKIP}
    fields['verbose'] = VERBOSE_TMPL % fields
    fields['verbose_optional'] = re.sub('(\\n\\s+validations:)?\\n\\s+required: true', '', fields['verbose'])
    (infile, outfile) = get_filename_args(has_infile=True)
    write_file(outfile, read_file(infile) % fields)
if __name__ == '__main__':
    main()