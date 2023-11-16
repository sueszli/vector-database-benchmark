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

Name: Upscaler
Author: K4YT3X <i@k4yt3x.com>
"""
import math
import time
from importlib import import_module
from PIL import Image
from .processor import Processor

class Upscaler:
    ALGORITHM_FIXED_SCALING_RATIOS = {'anime4k': [-1], 'realcugan': [1, 2, 3, 4], 'realsr': [4], 'srmd': [2, 3, 4], 'waifu2x': [1, 2]}
    ALGORITHM_CLASSES = {'anime4k': 'anime4k_python.Anime4K', 'realcugan': 'realcugan_ncnn_vulkan_python.Realcugan', 'realsr': 'realsr_ncnn_vulkan_python.Realsr', 'srmd': 'srmd_ncnn_vulkan_python.Srmd', 'waifu2x': 'waifu2x_ncnn_vulkan_python.Waifu2x'}
    processor_objects = {}

    @staticmethod
    def _get_scaling_tasks(input_width: int, input_height: int, output_width: int, output_height: int, algorithm: str) -> list:
        if False:
            i = 10
            return i + 15
        '\n        Get the required tasks for upscaling the image until it is larger than\n        or equal to the desired output dimensions. For example, SRMD only supports\n        2x, 3x, and 4x, so upsclaing an image from 320x240 to 3840x2160 will\n        require the SRMD to run 3x then 4x. In this case, this function will\n        return [3, 4].\n\n        :param input_width int: input image width\n        :param input_height int: input image height\n        :param output_width int: desired output image width\n        :param output_height int: desired output image size\n        :param algorithm str: upsclaing algorithm\n        :rtype list: the list of upsclaing tasks required\n        '
        output_scale = max(output_width / input_width, output_height / input_height)
        supported_scaling_ratios = sorted(Upscaler.ALGORITHM_FIXED_SCALING_RATIOS[algorithm])
        remaining_scaling_ratio = math.ceil(output_scale)
        if remaining_scaling_ratio == 1:
            return [supported_scaling_ratios[0]]
        if supported_scaling_ratios[0] == -1:
            return [remaining_scaling_ratio]
        scaling_jobs = []
        while remaining_scaling_ratio > 1:
            for ratio in supported_scaling_ratios:
                if ratio >= remaining_scaling_ratio:
                    scaling_jobs.append(ratio)
                    remaining_scaling_ratio /= ratio
                    break
            else:
                found = False
                for i in supported_scaling_ratios:
                    for j in supported_scaling_ratios:
                        if i * j >= remaining_scaling_ratio:
                            scaling_jobs.extend([i, j])
                            remaining_scaling_ratio /= i * j
                            found = True
                            break
                    if found is True:
                        break
                if found is False:
                    scaling_jobs.append(supported_scaling_ratios[-1])
                    remaining_scaling_ratio /= supported_scaling_ratios[-1]
        return scaling_jobs

    def upscale_image(self, image: Image.Image, output_width: int, output_height: int, algorithm: str, noise: int) -> Image.Image:
        if False:
            while True:
                i = 10
        '\n        upscale an image\n\n        :param image Image.Image: the image to upscale\n        :param output_width int: the desired output width\n        :param output_height int: the desired output height\n        :param algorithm str: the algorithm to use\n        :param noise int: the noise level (available only for some algorithms)\n        :rtype Image.Image: the upscaled image\n        '
        (width, height) = image.size
        for task in self._get_scaling_tasks(width, height, output_width, output_height, algorithm):
            processor_object = self.processor_objects.get((algorithm, task))
            if processor_object is None:
                (module_name, class_name) = self.ALGORITHM_CLASSES[algorithm].rsplit('.', 1)
                processor_module = import_module(module_name)
                processor_class = getattr(processor_module, class_name)
                processor_object = processor_class(noise=noise, scale=task)
                self.processor_objects[algorithm, task] = processor_object
            image = processor_object.process(image)
        return image.resize((output_width, output_height), Image.Resampling.LANCZOS)

class UpscalerProcessor(Processor, Upscaler):

    def process(self) -> None:
        if False:
            print('Hello World!')
        task = self.tasks_queue.get()
        while task is not None:
            try:
                if self.pause_flag.value is True:
                    time.sleep(0.1)
                    continue
                (frame_index, previous_frame, current_frame, (output_width, output_height, algorithm, noise, threshold)) = task
                difference_ratio = 0
                if previous_frame is not None:
                    difference_ratio = self.get_image_diff(previous_frame, current_frame)
                if difference_ratio < threshold:
                    self.processed_frames[frame_index] = True
                else:
                    self.processed_frames[frame_index] = self.upscale_image(current_frame, output_width, output_height, algorithm, noise)
                task = self.tasks_queue.get()
            except KeyboardInterrupt:
                break