"""
Copyright (C) 2018-2023 K4YT3X and contributors.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

__      __  _       _                  ___   __   __
\\ \\    / / (_)     | |                |__ \\  \\ \\ / /
 \\ \\  / /   _    __| |   ___    ___      ) |  \\ V /
  \\ \\/ /   | |  / _` |  / _ \\  / _ \\    / /    > <
   \\  /    | | | (_| | |  __/ | (_) |  / /_   / . \\
    \\/     |_|  \\__,_|  \\___|  \\___/  |____| /_/ \\_\\


Name: Video2X
Author: K4YT3X <i@k4yt3x.com>
Maintainer: BrianPetkovsek
Maintainer: SAT3LL
Maintainer: 28598519a
"""
import ctypes
import math
import signal
import sys
import time
from enum import Enum
from importlib import import_module
from multiprocessing import Manager, Pool, Queue, Value
from pathlib import Path
from typing import Callable, Optional
import ffmpeg
from cv2 import cv2
from loguru import logger
from rich.console import Console
from rich.file_proxy import FileProxy
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from video2x.processor import Processor
from . import __version__
from .decoder import VideoDecoder, VideoDecoderThread
from .encoder import VideoEncoder
from .interpolator import Interpolator, InterpolatorProcessor
from .upscaler import Upscaler, UpscalerProcessor
try:
    from pynput.keyboard import HotKey, Listener
except ImportError:
    ENABLE_HOTKEY = False
else:
    ENABLE_HOTKEY = True
LOGURU_FORMAT = '<green>{time:HH:mm:ss.SSSSSS!UTC}</green> | <level>{level: <8}</level> | <level>{message}</level>'

class ProcessingSpeedColumn(ProgressColumn):
    """Custom progress bar column that displays the processing speed"""

    def render(self, task: Task) -> Text:
        if False:
            i = 10
            return i + 15
        speed = task.finished_speed or task.speed
        return Text(f"{(round(speed, 2) if isinstance(speed, float) else '?')} FPS", style='progress.data.speed')

class ProcessingMode(Enum):
    UPSCALE = {'label': 'Upscaling', 'processor': UpscalerProcessor}
    INTERPOLATE = {'label': 'Interpolating', 'processor': InterpolatorProcessor}

class Video2X:
    """
    Video2X class

    provides two vital functions:
        - upscale: perform upscaling on a file
        - interpolate: perform motion interpolation on a file
    """

    def __init__(self, progress_callback: Optional[Callable]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.version = __version__
        self.progress_callback = progress_callback

    @staticmethod
    def _get_video_info(path: Path) -> tuple:
        if False:
            return 10
        "\n        get video file information with FFmpeg\n\n        :param path Path: video file path\n        :raises RuntimeError: raised when video stream isn't found\n        "
        logger.info('Reading input video information')
        for stream in ffmpeg.probe(path)['streams']:
            if stream['codec_type'] == 'video':
                video_info = stream
                break
        else:
            raise RuntimeError('unable to find video stream')
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError('OpenCV has failed to open the input file')
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        return (video_info['width'], video_info['height'], total_frames, frame_rate)

    def _run(self, input_path: Path, width: int, height: int, total_frames: int, frame_rate: float, output_path: Path, output_width: int, output_height: int, mode: ProcessingMode, processes: int, processing_settings: tuple) -> None:
        if False:
            for i in range(10):
                print('nop')
        if mode == ProcessingMode.UPSCALE:
            standalone_processor_path: str = Upscaler.ALGORITHM_CLASSES[processing_settings[2]]
            (module_name, class_name) = standalone_processor_path.rsplit('.', 1)
            processor_module = import_module(module_name)
            standalone_processor = getattr(processor_module, class_name)
            if getattr(standalone_processor, 'process', None) is None:
                logger.warning('No progress bar available for this processor')
                standalone_processor().process_video(input_path, output_path, width, height, output_width=output_width, output_height=output_height)
                return
        else:
            standalone_processor_path: str = Interpolator.ALGORITHM_CLASSES[processing_settings[1]]
            (module_name, class_name) = standalone_processor_path.rsplit('.', 1)
            processor_module = import_module(module_name)
            standalone_processor = getattr(processor_module, class_name)
            if getattr(standalone_processor, 'process', None) is None:
                logger.warning('No progress bar available for this processor')
                standalone_processor().process_video(input_path, output_path, frame_rate=frame_rate)
                return
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        console = Console()
        sys.stdout = FileProxy(console, sys.stdout)
        sys.stderr = FileProxy(console, sys.stderr)
        logger.remove()
        logger.add(sys.stderr, colorize=True, format=LOGURU_FORMAT)
        tasks_queue = Queue(maxsize=processes * 10)
        processed_frames = Manager().dict()
        pause_flag = Value(ctypes.c_bool, False)
        logger.info('Starting video decoder')
        decoder = VideoDecoder(input_path, width, height, frame_rate)
        decoder_thread = VideoDecoderThread(tasks_queue, decoder, processing_settings)
        decoder_thread.start()
        logger.info('Starting video encoder')
        encoder = VideoEncoder(input_path, frame_rate * 2 if mode == ProcessingMode.INTERPOLATE else frame_rate, output_path, output_width, output_height)
        processor: Processor = mode.value['processor'](tasks_queue, processed_frames, pause_flag)
        processor_pool = Pool(processes, processor.process)
        self.progress = Progress('[progress.description]{task.description}', BarColumn(complete_style='blue', finished_style='green'), '[progress.percentage]{task.percentage:>3.0f}%', '[color(240)]({task.completed}/{task.total})', ProcessingSpeedColumn(), TimeElapsedColumn(), '<', TimeRemainingColumn(), console=console, speed_estimate_period=300.0, disable=True)
        task = self.progress.add_task(f"[cyan]{mode.value['label']}", total=total_frames)

        def _toggle_pause(_signal_number: int=-1, _frame=None):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal pause_flag
            if pause_flag.value is False:
                self.progress.update(task, description=f"[cyan]{mode.value['label']} (paused)")
                self.progress.stop_task(task)
                logger.warning('Processing paused, press Ctrl+Alt+V again to resume')
            elif pause_flag.value is True:
                self.progress.update(task, description=f"[cyan]{mode.value['label']}")
                logger.warning('Resuming processing')
                self.progress.start_task(task)
            with pause_flag.get_lock():
                pause_flag.value = not pause_flag.value
        signal.signal(signal.SIGUSR1, _toggle_pause)
        if ENABLE_HOTKEY is True:
            pause_hotkey = HotKey(HotKey.parse('<ctrl>+<alt>+v'), _toggle_pause)
            keyboard_listener = Listener(on_press=lambda key: pause_hotkey.press(keyboard_listener.canonical(key)), on_release=lambda key: pause_hotkey.release(keyboard_listener.canonical(key)))
            keyboard_listener.start()
        exceptions = []
        try:
            with self.progress:
                frame_index = 0
                while frame_index < total_frames:
                    current_frame = processed_frames.get(frame_index)
                    if pause_flag.value is True or current_frame is None:
                        time.sleep(0.1)
                        continue
                    if frame_index == 0:
                        self.progress.disable = False
                        self.progress.start()
                    if current_frame is True:
                        encoder.write(processed_frames.get(frame_index - 1))
                    else:
                        encoder.write(current_frame)
                        if frame_index > 0:
                            del processed_frames[frame_index - 1]
                    self.progress.update(task, completed=frame_index + 1)
                    if self.progress_callback is not None:
                        self.progress_callback(frame_index + 1, total_frames)
                    frame_index += 1
        except (SystemExit, KeyboardInterrupt) as error:
            logger.warning('Exit signal received, exiting gracefully')
            logger.warning('Press ^C again to force terminate')
            exceptions.append(error)
        except Exception as error:
            logger.exception(error)
            exceptions.append(error)
        else:
            logger.info('Processing has completed')
            logger.info('Writing video trailer')
        finally:
            if ENABLE_HOTKEY is True:
                keyboard_listener.stop()
                keyboard_listener.join()
            if len(exceptions) > 0:
                decoder.kill()
                encoder.kill()
            decoder_thread.stop()
            decoder_thread.join()
            while tasks_queue.empty() is not True:
                tasks_queue.get()
            for _ in range(processes):
                tasks_queue.put(None)
            processor_pool.close()
            processor_pool.join()
            encoder.join()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logger.remove()
            logger.add(sys.stderr, colorize=True, format=LOGURU_FORMAT)
            if len(exceptions) > 0:
                raise exceptions[0]

    def upscale(self, input_path: Path, output_path: Path, output_width: int, output_height: int, noise: int, processes: int, threshold: float, algorithm: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        (width, height, total_frames, frame_rate) = self._get_video_info(input_path)
        if output_width == 0 or output_width is None:
            output_width = output_height / height * width
        elif output_height == 0 or output_width is None:
            output_height = output_width / width * height
        output_width = int(math.ceil(output_width / 2.0) * 2)
        output_height = int(math.ceil(output_height / 2.0) * 2)
        self._run(input_path, width, height, total_frames, frame_rate, output_path, output_width, output_height, ProcessingMode.UPSCALE, processes, (output_width, output_height, algorithm, noise, threshold))

    def interpolate(self, input_path: Path, output_path: Path, processes: int, threshold: float, algorithm: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        (width, height, original_frames, frame_rate) = self._get_video_info(input_path)
        total_frames = original_frames * 2 - 1
        self._run(input_path, width, height, total_frames, frame_rate, output_path, width, height, ProcessingMode.INTERPOLATE, processes, (threshold, algorithm))