from typing import List, NamedTuple, Optional, Tuple

class LlvmCoverageSegment(NamedTuple):
    line: int
    col: int
    segment_count: int
    has_count: int
    is_region_entry: int
    is_gap_entry: Optional[int]

    @property
    def has_coverage(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.segment_count > 0

    @property
    def is_executable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.has_count > 0

    def get_coverage(self, prev_segment: 'LlvmCoverageSegment') -> Tuple[List[int], List[int]]:
        if False:
            return 10
        if not prev_segment.is_executable:
            return ([], [])
        end_of_segment = self.line if self.col == 1 else self.line + 1
        lines_range = list(range(prev_segment.line, end_of_segment))
        return (lines_range, []) if prev_segment.has_coverage else ([], lines_range)

def parse_segments(raw_segments: List[List[int]]) -> List[LlvmCoverageSegment]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates LlvmCoverageSegment from a list of lists in llvm export json.\n    each segment is represented by 5-element array.\n    '
    ret: List[LlvmCoverageSegment] = []
    for raw_segment in raw_segments:
        assert len(raw_segment) == 5 or len(raw_segment) == 6, 'list is not compatible with llvmcom export:'
        ' Expected to have 5 or 6 elements'
        if len(raw_segment) == 5:
            ret.append(LlvmCoverageSegment(raw_segment[0], raw_segment[1], raw_segment[2], raw_segment[3], raw_segment[4], None))
        else:
            ret.append(LlvmCoverageSegment(*raw_segment))
    return ret