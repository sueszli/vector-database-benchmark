from copy import copy
from pathlib import Path
from typing import List

class ComputingSubtaskStateSnapshot:

    def __init__(self, *, subtask_id: str, progress: float, seconds_to_timeout: float, running_time_seconds: float, outfilebasename: str=None, output_format: str=None, scene_file: str=None, frames: List[int]=None, start_task: int=None, total_tasks: int=None, **_kwargs) -> None:
        if False:
            i = 10
            return i + 15
        self.subtask_id = subtask_id
        self.progress = progress
        self.seconds_to_timeout = seconds_to_timeout
        self.running_time_seconds = running_time_seconds
        self.outfilebasename = outfilebasename
        self.output_format = output_format
        self.scene_file = Path(scene_file).name if scene_file else None
        self.frames = copy(frames)
        self.start_task = start_task
        self.total_tasks = total_tasks

class LocalTaskStateSnapshot:

    def __init__(self, task_id, total_tasks, active_tasks, progress):
        if False:
            i = 10
            return i + 15
        self.task_id = task_id
        self.total_tasks = total_tasks
        self.active_tasks = active_tasks
        self.progress = progress

    def get_task_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self.task_id

    def get_total_tasks(self):
        if False:
            print('Hello World!')
        return self.total_tasks

    def get_active_tasks(self):
        if False:
            return 10
        return self.active_tasks

    def get_progress(self):
        if False:
            i = 10
            return i + 15
        return self.progress