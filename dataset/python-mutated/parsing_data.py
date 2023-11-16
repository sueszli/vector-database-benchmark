import os
from dataclasses import dataclass
from typing import Dict
from typing import Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from semgrep.core_targets_plan import Plan
import semgrep.semgrep_interfaces.semgrep_output_v1 as out
from semgrep.semgrep_types import Language

@dataclass
class LanguageParseData:
    targets_with_errors: int
    num_targets: int
    error_bytes: int
    num_bytes: int

class ParsingData:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._file_info: Dict[str, Tuple[Language, bool]] = {}
        self._parse_errors_by_lang: Dict[Language, LanguageParseData] = {}

    def add_targets(self, plan: 'Plan') -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds the targets from a given plan to the set of files tracked for\n        parsing statistics\n        '
        for task in plan.target_mappings:
            if not task.analyzer.definition.is_target_language:
                continue
            self._file_info[task.path] = (task.analyzer, True)
            entry = self._parse_errors_by_lang.get(task.analyzer, LanguageParseData(0, 0, 0, 0))
            try:
                entry.num_bytes += os.path.getsize(task.path)
                entry.num_targets += 1
            except OSError:
                pass
            self._parse_errors_by_lang[task.analyzer] = entry

    def add_error(self, err: out.CoreError) -> None:
        if False:
            while True:
                i = 10
        '\n        Records the targets/bytes which were not parsed as a result of the\n        given error. The file the error originated from should have been\n        registered from the original plan with `add_targets`.\n        '
        path = err.location.path.value
        try:
            (lang, no_error_yet) = self._file_info[path]
        except KeyError:
            return
        lang_parse_data = self._parse_errors_by_lang[lang]
        if no_error_yet:
            lang_parse_data.targets_with_errors += 1
        if isinstance(err.error_type.value, (out.LexicalError, out.ParseError, out.OtherParseError, out.AstBuilderError)):
            try:
                lang_parse_data.error_bytes += os.path.getsize(path)
            except OSError:
                pass
        elif isinstance(err.error_type.value, out.PartialParsing):
            for loc in err.error_type.value.value:
                lang_parse_data.error_bytes += loc.end.offset - loc.start.offset
        else:
            raise TypeError

    def get_errors_by_lang(self) -> Dict[Language, LanguageParseData]:
        if False:
            while True:
                i = 10
        return self._parse_errors_by_lang