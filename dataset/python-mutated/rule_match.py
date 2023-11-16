import binascii
import hashlib
import textwrap
from collections import Counter
from datetime import datetime
from functools import total_ordering
from pathlib import Path
from typing import Any
from typing import Counter as CounterType
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from uuid import UUID
from attrs import evolve
from attrs import field
from attrs import frozen
import semgrep.semgrep_interfaces.semgrep_output_v1 as out
from semgrep.constants import NOSEM_INLINE_COMMENT_RE
from semgrep.constants import RuleScanSource
from semgrep.external.pymmh3 import hash128
from semgrep.rule import Rule
from semgrep.semgrep_interfaces.semgrep_output_v1 import Direct
from semgrep.semgrep_interfaces.semgrep_output_v1 import Transitive
from semgrep.semgrep_interfaces.semgrep_output_v1 import Transitivity
from semgrep.util import get_lines
if TYPE_CHECKING:
    from semgrep.rule import Rule

def rstrip(value: Optional[str]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    return value.rstrip() if value is not None else None

@total_ordering
@frozen(eq=False)
class RuleMatch:
    """
    A section of code that matches a single rule (which is potentially many patterns).

    This is also often referred to as a finding.
    TODO: Rename this class to Finding?
    """
    match: out.CoreMatch
    message: str = field(repr=False)
    severity: out.MatchSeverity
    metadata: Dict[str, Any] = field(repr=False, factory=dict)
    extra: Dict[str, Any] = field(repr=False, factory=dict)
    fix: Optional[str] = field(converter=rstrip, default=None)
    fix_regex: Optional[out.FixRegex] = None
    index: int = 0
    match_based_index: int = 0
    match_formula_string: str = ''
    is_ignored: Optional[bool] = field(default=None)
    lines: List[str] = field(init=False, repr=False)
    previous_line: str = field(init=False, repr=False)
    syntactic_context: str = field(init=False, repr=False)
    cli_unique_key: Tuple = field(init=False, repr=False)
    ci_unique_key: Tuple = field(init=False, repr=False)
    ordering_key: Tuple = field(init=False, repr=False)
    match_based_key: Tuple = field(init=False, repr=False)
    syntactic_id: str = field(init=False, repr=False)
    match_based_id: str = field(init=False, repr=False)
    code_hash: str = field(init=False, repr=False)
    pattern_hash: str = field(init=False, repr=False)
    start_line_hash: str = field(init=False, repr=False)
    end_line_hash: str = field(init=False, repr=False)

    @property
    def rule_id(self) -> str:
        if False:
            while True:
                i = 10
        return self.match.check_id.value

    @property
    def path(self) -> Path:
        if False:
            i = 10
            return i + 15
        return Path(self.match.path.value)

    @property
    def start(self) -> out.Position:
        if False:
            while True:
                i = 10
        return self.match.start

    @property
    def end(self) -> out.Position:
        if False:
            print('Hello World!')
        return self.match.end

    @property
    def product(self) -> out.Product:
        if False:
            while True:
                i = 10
        if self.metadata.get('product') == 'secrets':
            return out.Product(out.Secrets())
        elif 'sca_info' in self.extra:
            return out.Product(out.SCA())
        else:
            return out.Product(out.SAST())

    @property
    def validation_state(self) -> Optional[out.ValidationState]:
        if False:
            for i in range(10):
                print('nop')
        return self.match.extra.validation_state

    @property
    def title(self) -> str:
        if False:
            return 10
        if isinstance(self.product.value, out.SCA):
            cve_id = self.metadata.get('sca-vuln-database-identifier')
            sca_info = self.extra.get('sca_info')
            package_name = sca_info.dependency_match.found_dependency.package if sca_info else None
            if cve_id and package_name:
                return f'{package_name} - {cve_id}'
        return self.rule_id

    def get_individual_line(self, line_number: int) -> str:
        if False:
            i = 10
            return i + 15
        line_array = get_lines(self.path, line_number, line_number)
        if len(line_array) == 0:
            return ''
        else:
            return line_array[0]

    @lines.default
    def get_lines(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return lines in file that this RuleMatch is referring to.\n\n        Assumes file exists.\n\n        Need to do on initialization instead of on read since file might not be the same\n        at read time\n        '
        return get_lines(self.path, self.start.line, self.end.line)

    @previous_line.default
    def get_previous_line(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the line preceding the match, if any.\n\n        This is meant for checking for the presence of a nosemgrep comment.\n        '
        return self.get_individual_line(self.start.line - 1) if self.start.line > 1 else ''

    @syntactic_context.default
    def get_syntactic_context(self) -> str:
        if False:
            return 10
        "\n        The code that matched, with whitespace and nosem comments removed.\n\n        This is useful to so that findings can be considered the same\n        when `    5 == 5` is updated to `  5 == 5  # nosemgrep`,\n        and thus CI systems don't retrigger notifications.\n        "
        lines = [*self.lines]
        if len(lines) > 0:
            lines[0] = NOSEM_INLINE_COMMENT_RE.sub('', lines[0])
            lines[0] = lines[0].rstrip() + '\n'
        code = ''.join(lines)
        code = textwrap.dedent(code)
        code = code.strip()
        return code

    @cli_unique_key.default
    def get_cli_unique_key(self) -> Tuple:
        if False:
            for i in range(10):
                print('nop')
        '\n        A unique key designed with data-completeness & correctness in mind.\n\n        Results in more unique findings than ci_unique_key.\n\n        Used for deduplication in the CLI before writing output.\n        '
        return (self.annotated_rule_name if self.from_transient_scan else self.rule_id, str(self.path), self.start.offset, self.end.offset, self.message, None, self.match.extra.validation_state)

    @ci_unique_key.default
    def get_ci_unique_key(self) -> Tuple:
        if False:
            while True:
                i = 10
        '\n        A unique key designed with notification user experience in mind.\n\n        Results in fewer unique findings than cli_unique_key.\n        '
        try:
            path = self.path.relative_to(Path.cwd())
        except (ValueError, FileNotFoundError):
            path = self.path
        if self.from_transient_scan:
            return (self.annotated_rule_name, str(path), self.syntactic_context, self.index)
        return (self.rule_id, str(path), self.syntactic_context, self.index)

    def get_path_changed_ci_unique_key(self, rename_dict: Dict[str, Path]) -> Tuple:
        if False:
            return 10
        '\n        A unique key that accounts for filepath renames.\n\n        Results in fewer unique findings than cli_unique_key.\n        '
        try:
            path = str(self.path.relative_to(Path.cwd()))
        except (ValueError, FileNotFoundError):
            path = str(self.path)
        renamed_path = str(rename_dict[path]) if path in rename_dict else path
        return (self.rule_id, renamed_path, self.syntactic_context, self.index)

    @ordering_key.default
    def get_ordering_key(self) -> Tuple:
        if False:
            print('Hello World!')
        '\n        Used to sort findings in output.\n\n        Note that we often batch by rule ID when gathering matches,\n        so the included self.rule_id will not do anything in those cases.\n\n        The message field is included to ensure a consistent ordering\n        when two findings match with different metavariables on the same code.\n        '
        return (self.path, self.start, self.end, self.rule_id, self.message)

    @syntactic_id.default
    def get_syntactic_id(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        A 32-character hash representation of ci_unique_key.\n\n        This value is sent to semgrep.dev and used to track findings across branches\n        '
        hash_int = hash128(str(self.ci_unique_key))
        hash_bytes = int.to_bytes(hash_int, byteorder='big', length=16, signed=False)
        return str(binascii.hexlify(hash_bytes), 'ascii')

    @match_based_key.default
    def get_match_based_key(self) -> Tuple:
        if False:
            for i in range(10):
                print('nop')
        "\n        A unique key with match based id's notion of uniqueness in mind.\n\n        We use this to check if two different findings will have the same match\n        based id or not. This is so we can then index them accordingly so two\n        similar findings will have unique match based IDs\n        "
        try:
            path = self.path.relative_to(Path.cwd())
        except (ValueError, FileNotFoundError):
            path = self.path
        match_formula_str = self.match_formula_string
        if self.extra.get('metavars') is not None:
            metavars = self.extra['metavars']
            for metavar in metavars:
                match_formula_str = match_formula_str.replace(metavar, metavars[metavar]['abstract_content'])
        if self.from_transient_scan:
            return (match_formula_str, path, self.annotated_rule_name)
        return (match_formula_str, path, self.rule_id)

    @match_based_id.default
    def get_match_based_id(self) -> str:
        if False:
            print('Hello World!')
        match_id = self.get_match_based_key()
        match_id_str = str(match_id)
        return f'{hashlib.blake2b(str.encode(match_id_str)).hexdigest()}_{str(self.match_based_index)}'

    @code_hash.default
    def get_code_hash(self) -> str:
        if False:
            return 10
        '\n        A 32-character hash representation of syntactic_context.\n\n        We started collecting this in addition to syntactic_id because the syntactic_id changes\n        whenever the file path or rule name or index changes, but all of those can be sent in\n        plaintext, so it does not make sense to include them inside a hash.\n\n        By sending the hash of ONLY the code contents, we can determine whether a finding\n        has moved files (path changed), or moved down a file (index changed) and handle that\n        logic in the app. This also gives us more flexibility for changing the definition of\n        a unique finding in the future, since we can analyze things like code, file, index all\n        independently of one another.\n        '
        return hashlib.sha256(self.syntactic_context.encode()).hexdigest()

    @pattern_hash.default
    def get_pattern_hash(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        A 32-character hash representation of syntactic_context.\n\n        We started collecting this in addition to match_based_id because the match_based_id will\n        change when the file path or rule name or index changes, but all of those can be passed\n        in plaintext, so it does not make sense to include them in the hash.\n\n        By sending the hash of ONLY the pattern contents, we can determine whether a finding\n        has only changed because e.g. the file path changed\n        '
        match_formula_str = self.match_formula_string
        if self.extra.get('metavars') is not None:
            metavars = self.extra['metavars']
            for metavar in metavars:
                match_formula_str = match_formula_str.replace(metavar, metavars[metavar]['abstract_content'])
        return hashlib.sha256(match_formula_str.encode()).hexdigest()

    @start_line_hash.default
    def get_start_line_hash(self) -> str:
        if False:
            while True:
                i = 10
        '\n        A 32-character hash of the first line of the code in the match\n        '
        first_line = self.get_individual_line(self.start.line)
        return hashlib.sha256(first_line.encode()).hexdigest()

    @end_line_hash.default
    def get_end_line_hash(self) -> str:
        if False:
            while True:
                i = 10
        '\n        A 32-character hash of the last line of the code in the match\n        '
        last_line = self.get_individual_line(self.end.line)
        return hashlib.sha256(last_line.encode()).hexdigest()

    @property
    def is_sca_match_in_direct_dependency(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return 'sca_info' in self.extra and self.extra['sca_info'].dependency_match.found_dependency.transitivity == Transitivity(Direct())

    @property
    def is_sca_match_in_transitive_dependency(self) -> bool:
        if False:
            i = 10
            return i + 15
        return 'sca_info' in self.extra and self.extra['sca_info'].dependency_match.found_dependency.transitivity == Transitivity(Transitive())

    @property
    def is_reachable_in_code_sca_match(self) -> bool:
        if False:
            while True:
                i = 10
        return 'sca_info' in self.extra and self.extra['sca_info'].reachable

    @property
    def is_always_reachable_sca_match(self) -> bool:
        if False:
            print('Hello World!')
        return 'sca-kind' in self.metadata and self.metadata['sca-kind'] == 'upgrade-only'

    @property
    def uuid(self) -> UUID:
        if False:
            i = 10
            return i + 15
        '\n        A UUID representation of ci_unique_key.\n        '
        return UUID(hex=self.syntactic_id)

    @property
    def is_blocking(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns if this finding indicates it should block CI\n        '
        blocking = 'block' in self.metadata.get('dev.semgrep.actions', ['block'])
        if 'sca_info' in self.extra:
            if self.is_always_reachable_sca_match and self.is_sca_match_in_transitive_dependency or not self.exposure_type == 'reachable':
                return False
            else:
                return blocking
        else:
            return blocking

    @property
    def dataflow_trace(self) -> Optional[out.MatchDataflowTrace]:
        if False:
            return 10
        return self.match.extra.dataflow_trace

    @property
    def exposure_type(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        Mimic the exposure categories on semgrep.dev for supply chain.\n\n        "reachable": dependency is used in the codebase or is vulnerable even without usage\n        "unreachable": dependency is not used in the codebase\n        "undetermined": rule for dependency doesn\'t look for reachability\n        None: not a supply chain rule\n        '
        if 'sca_info' not in self.extra:
            return None
        if self.metadata.get('sca-kind') == 'upgrade-only':
            return 'reachable'
        elif self.metadata.get('sca-kind') == 'legacy':
            return 'undetermined'
        else:
            return 'reachable' if self.extra['sca_info'].reachable else 'unreachable'

    def to_app_finding_format(self, commit_date: str) -> out.Finding:
        if False:
            for i in range(10):
                print('nop')
        '\n        commit_date here for legacy reasons.\n        commit date of the head commit in epoch time\n        '
        commit_date_app_format = datetime.fromtimestamp(int(commit_date)).isoformat()
        if isinstance(self.severity.value, out.Error):
            app_severity = 2
        elif isinstance(self.severity.value, out.Warning):
            app_severity = 1
        elif isinstance(self.severity.value, out.Experiment):
            app_severity = 4
        else:
            app_severity = 0
        hashes = out.FindingHashes(start_line_hash=self.start_line_hash, end_line_hash=self.end_line_hash, code_hash=self.code_hash, pattern_hash=self.pattern_hash)
        ret = out.Finding(check_id=out.RuleId(self.rule_id), path=out.Fpath(str(self.path)), line=self.start.line, column=self.start.col, end_line=self.end.line, end_column=self.end.col, message=self.message, severity=app_severity, index=self.index, commit_date=commit_date_app_format, syntactic_id=self.syntactic_id, match_based_id=self.match_based_id, hashes=hashes, metadata=out.RawJson(self.metadata), is_blocking=self.is_blocking, dataflow_trace=self.dataflow_trace, validation_state=self.match.extra.validation_state)
        if self.extra.get('fixed_lines'):
            ret.fixed_lines = self.extra.get('fixed_lines')
        if 'sca_info' in self.extra:
            ret.sca_info = self.extra['sca_info']
        return ret

    @property
    def scan_source(self) -> RuleScanSource:
        if False:
            while True:
                i = 10
        src: str = self.metadata.get('semgrep.dev', {}).get('src', '')
        if src == 'unchanged':
            return RuleScanSource.unchanged
        elif src == 'new-version':
            return RuleScanSource.new_version
        elif src == 'new-rule':
            return RuleScanSource.new_rule
        elif src == 'previous-scan':
            return RuleScanSource.previous_scan
        else:
            return RuleScanSource.unannotated

    @property
    def from_transient_scan(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.scan_source == RuleScanSource.previous_scan

    @property
    def annotated_rule_name(self) -> str:
        if False:
            return 10
        return self.metadata.get('semgrep.dev', {}).get('rule', {}).get('rule_name')

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        '\n        We use the "data-correctness" key to prevent keeping around duplicates.\n        '
        return hash(self.cli_unique_key)

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, type(self)):
            return False
        return self.cli_unique_key == other.cli_unique_key

    def __lt__(self, other: 'RuleMatch') -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.ordering_key < other.ordering_key

class RuleMatchSet(Iterable[RuleMatch]):
    """
    A custom set type which is aware when findings are the same.

    It also automagically adds the correct finding index when adding elements.
    """

    def __init__(self, rule: Rule, __iterable: Optional[Iterable[RuleMatch]]=None) -> None:
        if False:
            return 10
        self._match_based_counts: CounterType[Tuple] = Counter()
        self._ci_key_counts: CounterType[Tuple] = Counter()
        self._rule = rule
        if __iterable is None:
            self._set = set()
        else:
            self._set = set(__iterable)

    def add(self, match: RuleMatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Add finding, but if the same (rule, path, code) exists,\n        note this by incrementing the finding's index.\n\n        The index lets us still notify when some code with findings is duplicated,\n        even though we'd otherwise deduplicate the findings.\n        "
        if match.rule_id != self._rule.id:
            raise ValueError('Added match must have identical rule id to set rule')
        match = evolve(match, match_formula_string=self._rule.formula_string)
        self._match_based_counts[match.get_match_based_key()] += 1
        self._ci_key_counts[match.ci_unique_key] += 1
        match = evolve(match, index=self._ci_key_counts[match.ci_unique_key] - 1)
        match = evolve(match, match_based_index=self._match_based_counts[match.get_match_based_key()] - 1)
        self._set.add(match)

    def update(self, *rule_match_iterables: Iterable[RuleMatch]) -> None:
        if False:
            print('Hello World!')
        "\n        Add findings, but if the same (rule, path, code) exists,\n        note this by incrementing the finding's index.\n\n        The index lets us still notify when some code with findings is duplicated,\n        even though we'd otherwise deduplicate the findings.\n        "
        for rule_matches in rule_match_iterables:
            for rule_match in rule_matches:
                self.add(rule_match)

    def __iter__(self) -> Iterator[RuleMatch]:
        if False:
            return 10
        return iter(self._set)
OrderedRuleMatchList = List[RuleMatch]
RuleMatchMap = Dict['Rule', OrderedRuleMatchList]