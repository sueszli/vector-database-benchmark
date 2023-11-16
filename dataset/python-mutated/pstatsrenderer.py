from __future__ import annotations
import marshal
from typing import Any, Dict, Tuple
from pyinstrument import processors
from pyinstrument.frame import Frame
from pyinstrument.renderers.base import FrameRenderer, ProcessorList
from pyinstrument.session import Session
FrameKey = Tuple[str, int, str]
CallerValue = Tuple[float, int, float, float]
FrameValue = Tuple[float, int, float, float, Dict[FrameKey, CallerValue]]
StatsDict = Dict[FrameKey, FrameValue]

class PstatsRenderer(FrameRenderer):
    """
    Outputs a marshaled dict, containing processed frames in pstat format,
    suitable for processing by gprof2dot and snakeviz.
    """
    output_file_extension = 'pstats'
    output_is_binary = True

    def __init__(self, **kwargs: Any):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)

    def frame_key(self, frame: Frame) -> FrameKey:
        if False:
            print('Hello World!')
        return (frame.file_path or '', frame.line_no or 0, frame.function)

    def render_frame(self, frame: Frame | None, stats: StatsDict) -> None:
        if False:
            i = 10
            return i + 15
        if frame is None:
            return
        key = self.frame_key(frame)
        if key not in stats:
            call_time = -1
            number_calls = -1
            total_time = 0
            cumulative_time = 0
            callers: dict[FrameKey, CallerValue] = {}
        else:
            (call_time, number_calls, total_time, cumulative_time, callers) = stats[key]
        total_time += frame.total_self_time
        cumulative_time += frame.time
        if frame.parent:
            parent_key = self.frame_key(frame.parent)
            if parent_key not in callers:
                p_call_time = -1
                p_number_calls = -1
                p_total_time = 0
                p_cumulative_time = 0
            else:
                (p_call_time, p_number_calls, p_total_time, p_cumulative_time) = callers[parent_key]
            p_total_time += frame.total_self_time
            p_cumulative_time += frame.time
            callers[parent_key] = (p_call_time, p_number_calls, p_total_time, p_cumulative_time)
        stats[key] = (call_time, number_calls, total_time, cumulative_time, callers)
        for child in frame.children:
            if not frame.is_synthetic:
                self.render_frame(child, stats)

    def render(self, session: Session):
        if False:
            print('Hello World!')
        frame = self.preprocess(session.root_frame())
        stats = {}
        self.render_frame(frame, stats)
        return marshal.dumps(stats).decode(encoding='utf-8', errors='surrogateescape')

    def default_processors(self) -> ProcessorList:
        if False:
            while True:
                i = 10
        return [processors.remove_importlib, processors.remove_tracebackhide, processors.merge_consecutive_self_time, processors.aggregate_repeated_calls, processors.remove_unnecessary_self_time_nodes, processors.remove_irrelevant_nodes, processors.remove_first_pyinstrument_frames_processor]