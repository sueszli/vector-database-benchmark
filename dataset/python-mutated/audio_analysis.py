from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from fnmatch import fnmatch as _fnmatch
from glob import glob as _glob
import os as _os
from random import shuffle as _shuffle
import turicreate as _tc
from turicreate.toolkits._main import ToolkitError as _ToolkitError
from turicreate._deps.minimal_package import _minimal_package_import_check

def load_audio(path, with_path=True, recursive=True, ignore_failure=True, random_order=False):
    if False:
        i = 10
        return i + 15
    '\n    Loads WAV file(s) from a path.\n\n    Parameters\n    ----------\n    path : str\n        Path to WAV files to be loaded.\n\n    with_path : bool, optional\n        Indicates whether a path column is added to the returned SFrame.\n\n    recursive : bool, optional\n        Indicates whether ``load_audio`` should do a recursive directory traversal,\n        or only load audio files directly under ``path``.\n\n    ignore_failure : bool, optional\n        If True, only print warnings for failed files and keep loading the remaining\n        audio files.\n\n    random_order : bool, optional\n        Load audio files in random order.\n\n    Returns\n    -------\n    out : SFrame\n        Returns an SFrame with either an \'audio\' column or both an \'audio\' and\n        a \'path\' column. The \'audio\' column is a column of dictionaries.\n\n        Each dictionary contains two items. One item is the sample rate, in\n        samples per second (int type). The other item will be the data in a numpy\n        array. If the wav file has a single channel, the array will have a single\n        dimension. If there are multiple channels, the array will have shape\n        (L,C) where L is the number of samples and C is the number of channels.\n\n    Examples\n    --------\n    >>> audio_path = "~/Documents/myAudioFiles/"\n    >>> audio_sframe = tc.audio_analysis.load_audio(audio_path, recursive=True)\n    '
    _wavfile = _minimal_package_import_check('scipy.io.wavfile')
    path = _tc.util._make_internal_url(path)
    all_wav_files = []
    if _fnmatch(path, '*.wav'):
        all_wav_files.append(path)
    elif recursive:
        for (dir_path, _, file_names) in _os.walk(path):
            for cur_file in file_names:
                if _fnmatch(cur_file, '*.wav'):
                    all_wav_files.append(dir_path + '/' + cur_file)
    else:
        all_wav_files = _glob(path + '/*.wav')
    if random_order:
        _shuffle(all_wav_files)
    result_builder = _tc.SFrameBuilder(column_types=[dict, str], column_names=['audio', 'path'])
    for cur_file_path in all_wav_files:
        try:
            (sample_rate, data) = _wavfile.read(cur_file_path)
        except Exception as e:
            error_string = 'Could not read {}: {}'.format(cur_file_path, e)
            if not ignore_failure:
                raise _ToolkitError(error_string)
            else:
                print(error_string)
                continue
        result_builder.append([{'sample_rate': sample_rate, 'data': data}, cur_file_path])
    result = result_builder.close()
    if not with_path:
        del result['path']
    return result