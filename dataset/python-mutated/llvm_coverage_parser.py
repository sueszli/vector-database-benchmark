from typing import Any, Dict, List, Set, Tuple
from .coverage_record import CoverageRecord
from .llvm_coverage_segment import LlvmCoverageSegment, parse_segments

class LlvmCoverageParser:
    """
    Accepts a parsed json produced by llvm-cov export -- typically,
    representing a single C++ test and produces a list
    of CoverageRecord(s).

    """

    def __init__(self, llvm_coverage: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        self._llvm_coverage = llvm_coverage

    @staticmethod
    def _skip_coverage(path: str) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns True if file path should not be processed.\n        This is repo-specific and only makes sense for the current state of\n        ovrsource.\n        '
        if '/third-party/' in path:
            return True
        return False

    @staticmethod
    def _collect_coverage(segments: List[LlvmCoverageSegment]) -> Tuple[List[int], List[int]]:
        if False:
            i = 10
            return i + 15
        '\n        Stateful parsing of coverage segments.\n        '
        covered_lines: Set[int] = set()
        uncovered_lines: Set[int] = set()
        prev_segment = LlvmCoverageSegment(1, 0, 0, 0, 0, None)
        for segment in segments:
            (covered_range, uncovered_range) = segment.get_coverage(prev_segment)
            covered_lines.update(covered_range)
            uncovered_lines.update(uncovered_range)
            prev_segment = segment
        uncovered_lines.difference_update(covered_lines)
        return (sorted(covered_lines), sorted(uncovered_lines))

    def parse(self, repo_name: str) -> List[CoverageRecord]:
        if False:
            print('Hello World!')
        records: List[CoverageRecord] = []
        for export_unit in self._llvm_coverage['data']:
            for file_info in export_unit['files']:
                filepath = file_info['filename']
                if self._skip_coverage(filepath):
                    continue
                if filepath is None:
                    continue
                segments = file_info['segments']
                (covered_lines, uncovered_lines) = self._collect_coverage(parse_segments(segments))
                records.append(CoverageRecord(filepath, covered_lines, uncovered_lines))
        return records