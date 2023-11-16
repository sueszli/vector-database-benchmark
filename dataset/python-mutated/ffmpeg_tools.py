"""Miscellaneous bindings to ffmpeg."""
import os
from moviepy.config import FFMPEG_BINARY
from moviepy.decorators import convert_parameter_to_seconds, convert_path_to_string
from moviepy.tools import subprocess_call

@convert_path_to_string(('inputfile', 'outputfile'))
@convert_parameter_to_seconds(('start_time', 'end_time'))
def ffmpeg_extract_subclip(inputfile, start_time, end_time, outputfile=None, logger='bar'):
    if False:
        for i in range(10):
            print('nop')
    'Makes a new video file playing video file between two times.\n\n    Parameters\n    ----------\n\n    inputfile : str\n      Path to the file from which the subclip will be extracted.\n\n    start_time : float\n      Moment of the input clip that marks the start of the produced subclip.\n\n    end_time : float\n      Moment of the input clip that marks the end of the produced subclip.\n\n    outputfile : str, optional\n      Path to the output file. Defaults to\n      ``<inputfile_name>SUB<start_time>_<end_time><ext>``.\n    '
    if not outputfile:
        (name, ext) = os.path.splitext(inputfile)
        (t1, t2) = [int(1000 * t) for t in [start_time, end_time]]
        outputfile = '%sSUB%d_%d%s' % (name, t1, t2, ext)
    cmd = [FFMPEG_BINARY, '-y', '-ss', '%0.2f' % start_time, '-i', inputfile, '-t', '%0.2f' % (end_time - start_time), '-map', '0', '-vcodec', 'copy', '-acodec', 'copy', '-copyts', outputfile]
    subprocess_call(cmd, logger=logger)

@convert_path_to_string(('videofile', 'audiofile', 'outputfile'))
def ffmpeg_merge_video_audio(videofile, audiofile, outputfile, video_codec='copy', audio_codec='copy', logger='bar'):
    if False:
        return 10
    'Merges video file and audio file into one movie file.\n\n    Parameters\n    ----------\n\n    videofile : str\n      Path to the video file used in the merge.\n\n    audiofile : str\n      Path to the audio file used in the merge.\n\n    outputfile : str\n      Path to the output file.\n\n    video_codec : str, optional\n      Video codec used by FFmpeg in the merge.\n\n    audio_codec : str, optional\n      Audio codec used by FFmpeg in the merge.\n    '
    cmd = [FFMPEG_BINARY, '-y', '-i', audiofile, '-i', videofile, '-vcodec', video_codec, '-acodec', audio_codec, outputfile]
    subprocess_call(cmd, logger=logger)

@convert_path_to_string(('inputfile', 'outputfile'))
def ffmpeg_extract_audio(inputfile, outputfile, bitrate=3000, fps=44100, logger='bar'):
    if False:
        i = 10
        return i + 15
    'Extract the sound from a video file and save it in ``outputfile``.\n\n    Parameters\n    ----------\n\n    inputfile : str\n      The path to the file from which the audio will be extracted.\n\n    outputfile : str\n      The path to the file to which the audio will be stored.\n\n    bitrate : int, optional\n      Bitrate for the new audio file.\n\n    fps : int, optional\n      Frame rate for the new audio file.\n    '
    cmd = [FFMPEG_BINARY, '-y', '-i', inputfile, '-ab', '%dk' % bitrate, '-ar', '%d' % fps, outputfile]
    subprocess_call(cmd, logger=logger)

@convert_path_to_string(('inputfile', 'outputfile'))
def ffmpeg_resize(inputfile, outputfile, size, logger='bar'):
    if False:
        print('Hello World!')
    'Resizes a file to new size and write the result in another.\n\n    Parameters\n    ----------\n\n    inputfile : str\n      Path to the file to be resized.\n\n    outputfile : str\n      Path to the output file.\n\n    size : list or tuple\n      New size in format ``[width, height]`` for the output file.\n    '
    cmd = [FFMPEG_BINARY, '-i', inputfile, '-vf', 'scale=%d:%d' % (size[0], size[1]), outputfile]
    subprocess_call(cmd, logger=logger)

@convert_path_to_string(('inputfile', 'outputfile', 'output_dir'))
def ffmpeg_stabilize_video(inputfile, outputfile=None, output_dir='', overwrite_file=True, logger='bar'):
    if False:
        print('Hello World!')
    "\n    Stabilizes ``filename`` and write the result to ``output``.\n\n    Parameters\n    ----------\n\n    inputfile : str\n      The name of the shaky video.\n\n    outputfile : str, optional\n      The name of new stabilized video. Defaults to appending '_stabilized' to\n      the input file name.\n\n    output_dir : str, optional\n      The directory to place the output video in. Defaults to the current\n      working directory.\n\n    overwrite_file : bool, optional\n      If ``outputfile`` already exists in ``output_dir``, then overwrite\n      ``outputfile`` Defaults to True.\n    "
    if not outputfile:
        without_dir = os.path.basename(inputfile)
        (name, ext) = os.path.splitext(without_dir)
        outputfile = f'{name}_stabilized{ext}'
    outputfile = os.path.join(output_dir, outputfile)
    cmd = [FFMPEG_BINARY, '-i', inputfile, '-vf', 'deshake', outputfile]
    if overwrite_file:
        cmd.append('-y')
    subprocess_call(cmd, logger=logger)