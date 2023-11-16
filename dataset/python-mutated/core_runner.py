import asyncio
import collections
import contextlib
import json
import resource
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Coroutine
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from attr import evolve
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import TaskID
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from ruamel.yaml import YAML
import semgrep.semgrep_interfaces.semgrep_output_v1 as out
from semgrep.app import auth
from semgrep.config_resolver import Config
from semgrep.console import console
from semgrep.constants import Colors
from semgrep.constants import PLEASE_FILE_ISSUE_TEXT
from semgrep.core_output import core_error_to_semgrep_error
from semgrep.core_output import core_matches_to_rule_matches
from semgrep.core_targets_plan import Plan
from semgrep.core_targets_plan import Task
from semgrep.engine import EngineType
from semgrep.error import SemgrepCoreError
from semgrep.error import SemgrepError
from semgrep.error import with_color
from semgrep.output_extra import OutputExtra
from semgrep.parsing_data import ParsingData
from semgrep.rule import Rule
from semgrep.rule_match import OrderedRuleMatchList
from semgrep.rule_match import RuleMatchMap
from semgrep.semgrep_types import Language
from semgrep.state import DesignTreatment
from semgrep.state import get_state
from semgrep.target_manager import TargetManager
from semgrep.target_mode import TargetModeConfig
from semgrep.verbose_logging import getLogger
logger = getLogger(__name__)
INPUT_BUFFER_LIMIT: int = 1024 * 1024 * 1024
LARGE_READ_SIZE: int = 1024 * 1024 * 512

def get_contributions(engine_type: EngineType) -> out.Contributions:
    if False:
        return 10
    binary_path = engine_type.get_binary_path()
    start = datetime.now()
    if binary_path is None:
        raise SemgrepError('semgrep engine not found.')
    cmd = [str(binary_path), '-json', '-dump_contributions']
    env = get_state().env
    try:
        raw_output = subprocess.run(cmd, timeout=env.git_command_timeout, capture_output=True, encoding='utf-8', check=True).stdout
        contributions = out.Contributions.from_json_string(raw_output)
    except subprocess.CalledProcessError:
        logger.warning('Failed to collect contributions. Continuing with scan...')
        contributions = out.Contributions([])
    logger.debug(f'semgrep contributions ran in {datetime.now() - start}')
    return contributions

def setrlimits_preexec_fn() -> None:
    if False:
        return 10
    '\n    Sets stack limit of current running process to the maximum possible\n    of the following as allowed by the OS:\n    - 5120000\n    - stack hard limit / 3\n    - stack hard limit / 4\n    - current existing soft limit\n\n    Note this is intended to run as a preexec_fn before semgrep-core in a subprocess\n    so all code here runs in a child fork before os switches to semgrep-core binary\n    '
    core_logger = getLogger('semgrep_core')
    (old_soft_limit, hard_limit) = resource.getrlimit(resource.RLIMIT_STACK)
    core_logger.info(f'Existing stack limits: Soft: {old_soft_limit}, Hard: {hard_limit}')
    potential_soft_limits = [int(hard_limit / 3), int(hard_limit / 4), old_soft_limit * 100, old_soft_limit * 10, old_soft_limit * 5, 1000000000, 512000000, 51200000, 5120000, old_soft_limit]
    potential_soft_limits.sort(reverse=True)
    for soft_limit in potential_soft_limits:
        try:
            core_logger.info(f'Trying to set soft limit to {soft_limit}')
            resource.setrlimit(resource.RLIMIT_STACK, (soft_limit, hard_limit))
            core_logger.info(f'Successfully set stack limit to {soft_limit}, {hard_limit}')
            return
        except Exception as e:
            core_logger.info(f'Failed to set stack limit to {soft_limit}, {hard_limit}. Trying again.')
            core_logger.verbose(str(e))
    core_logger.info('Failed to change stack limits')

def dedup_errors(errors: List[SemgrepCoreError]) -> List[SemgrepCoreError]:
    if False:
        print('Hello World!')
    return list({uniq_error_id(e): e for e in errors}.values())

def uniq_error_id(error: SemgrepCoreError) -> Tuple[int, Path, out.Position, out.Position, str]:
    if False:
        return 10
    return (error.code, Path(error.core.location.path.value), error.core.location.start, error.core.location.end, error.core.message)

def open_and_ignore(fname: str) -> None:
    if False:
        print('Hello World!')
    "\n    Attempt to open 'fname' simply so a record of having done so will\n    be seen by 'strace'.\n    "
    try:
        with open(fname, 'rb') as in_file:
            pass
    except BaseException:
        pass

class StreamingSemgrepCore:
    """
    Handles running semgrep-core in a streaming fashion

    This behavior is assumed to be that semgrep-core:
    - prints a "." on a newline for every file it finishes scanning
    - prints a number on a newline for any extra targets produced during a scan
    - prints a single json blob of all results

    Exposes the subprocess.CompletedProcess properties for
    expediency in integrating
    """

    def __init__(self, cmd: List[str], total: int, engine_type: EngineType) -> None:
        if False:
            i = 10
            return i + 15
        '\n        cmd: semgrep-core command to run\n        total: how many rules to run / how many "." we expect to see a priori\n               used to display progress_bar\n        '
        self._cmd = cmd
        self._total = total
        self._stdout = ''
        self._stderr = ''
        self._progress_bar: Optional[Progress] = None
        self._progress_bar_task_id: Optional[TaskID] = None
        self._engine_type: EngineType = engine_type
        self.vfs_map: Dict[str, bytes] = {}

    @property
    def stdout(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._stdout

    @property
    def stderr(self) -> str:
        if False:
            while True:
                i = 10
        return self._stderr

    async def _core_stdout_processor(self, stream: asyncio.StreamReader) -> None:
        """
        Asynchronously process stdout of semgrep-core

        Updates progress bar one increment for every "." it sees from semgrep-core
        stdout

        Increases the progress bar total for any number reported from semgrep-core
        stdout

        When it sees neither output it saves it to self._stdout
        """
        stdout_lines: List[bytes] = []
        num_total_targets: int = self._total
        get_input: Callable[[asyncio.StreamReader], Coroutine[Any, Any, bytes]] = lambda s: s.readexactly(2)
        reading_json = False
        has_started = False
        while True:
            try:
                line_bytes = await get_input(stream)
            except asyncio.IncompleteReadError:
                logger.debug(self._stderr)
                raise SemgrepError(f'\n                    You are seeing this because the engine was killed.\n\n                    The most common reason this happens is because it used too much memory.\n                    If your repo is large (~10k files or more), you have three options:\n                    1. Increase the amount of memory available to semgrep\n                    2. Reduce the number of jobs semgrep runs with via `-j <jobs>`. We\n                        recommend using 1 job if you are running out of memory.\n                    3. Scan the repo in parts (contact us for help)\n\n                    Otherwise, it is likely that semgrep is hitting the limit on only some\n                    files. In this case, you can try to set the limit on the amount of memory\n                    semgrep can use on each file with `--max-memory <memory>`. We recommend\n                    lowering this to a limit 70% of the available memory. For CI runs with\n                    interfile analysis, the default max-memory is 5000MB. Without, the default\n                    is unlimited.\n\n                    The last thing you can try if none of these work is to raise the stack\n                    limit with `ulimit -s <limit>`.\n\n                    If you have tried all these steps and still are seeing this error, please\n                    contact us.\n\n                       Error: semgrep-core exited with unexpected output\n                    ')
            if not has_started and self._progress_bar and (self._progress_bar_task_id is not None):
                has_started = True
                self._progress_bar.start_task(self._progress_bar_task_id)
            if not line_bytes:
                self._stdout = b''.join(stdout_lines).decode('utf-8', 'replace')
                break
            if line_bytes == b'.\n' and (not reading_json):
                advanced_targets = 1 if self._engine_type.is_interfile else 3
                if self._progress_bar and self._progress_bar_task_id is not None:
                    self._progress_bar.update(self._progress_bar_task_id, advance=advanced_targets)
            elif chr(line_bytes[0]).isdigit() and (not reading_json):
                if not line_bytes.endswith(b'\n'):
                    line_bytes = line_bytes + await stream.readline()
                extra_targets = int(line_bytes)
                num_total_targets += extra_targets
                if self._progress_bar and self._progress_bar_task_id is not None:
                    self._progress_bar.update(self._progress_bar_task_id, total=num_total_targets)
            else:
                stdout_lines.append(line_bytes)
                reading_json = True
                get_input = lambda s: s.read(n=LARGE_READ_SIZE)

    async def _core_stderr_processor(self, stream: Optional[asyncio.StreamReader]) -> None:
        """
        Asynchronously process stderr of semgrep-core

        Basically works synchronously and combines output to
        stderr to self._stderr
        """
        stderr_lines: List[str] = []
        if stream is None:
            raise RuntimeError('subprocess was created without a stream')
        while True:
            line_bytes = await stream.readline()
            if not line_bytes:
                self._stderr = ''.join(stderr_lines)
                break
            line = line_bytes.decode('utf-8', 'replace')
            stderr_lines.append(line)

    def _handle_read_file(self, fname: str) -> Tuple[bytes, int]:
        if False:
            while True:
                i = 10
        "\n        Handler for semgrep_analyze 'read_file' callback.\n        "
        try:
            if fname in self.vfs_map:
                contents = self.vfs_map[fname]
                logger.debug(f'read_file: in memory {fname}: {len(contents)} bytes')
                return (contents, 0)
            with open(fname, 'rb') as in_file:
                contents = in_file.read()
                logger.debug(f'read_file: disk read {fname}: {len(contents)} bytes')
                return (contents, 0)
        except BaseException as e:
            logger.debug(f'read_file: reading {fname}: exn: {e!r}')
            exnClass = type(e).__name__
            return (f'{fname}: {exnClass}: {e}'.encode(), 1)

    async def _handle_process_outputs(self, stdout: asyncio.StreamReader, stderr: asyncio.StreamReader) -> None:
        """
        Wait for both output streams to reach EOF, processing and
        accumulating the results in the meantime.
        """
        results = await asyncio.gather(self._core_stdout_processor(stdout), self._core_stderr_processor(stderr), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                raise SemgrepError(f'Error while running rules: {r}')

    async def _stream_exec_subprocess(self) -> int:
        """
        Run semgrep-core via fork/exec, consuming its output
        asynchronously.

        Return its exit code when it terminates.
        """
        process = await asyncio.create_subprocess_exec(*self._cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=INPUT_BUFFER_LIMIT, preexec_fn=setrlimits_preexec_fn)
        assert process.stdout
        assert process.stderr
        await self._handle_process_outputs(process.stdout, process.stderr)
        return await process.wait()

    def execute(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Run semgrep-core and listen to stdout to update\n        progress_bar as necessary\n\n        Blocks til completion and returns exit code\n        '
        open_and_ignore('/tmp/core-runner-semgrep-BEGIN')
        terminal = get_state().terminal
        with Progress(TextColumn(' '), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console, disable=not sys.stderr.isatty() or self._total <= 1 or terminal.is_quiet or terminal.is_debug) as progress_bar:
            self._progress_bar = progress_bar
            self._progress_bar_task_id = self._progress_bar.add_task('', total=self._total, start=False)
            rc = asyncio.run(self._stream_exec_subprocess())
        open_and_ignore('/tmp/core-runner-semgrep-END')
        return rc

class CoreRunner:
    """
    Handles interactions between semgrep and semgrep-core

    This includes properly invoking semgrep-core and parsing the output
    """

    def __init__(self, jobs: Optional[int], engine_type: EngineType, timeout: int, max_memory: int, timeout_threshold: int, interfile_timeout: int, optimizations: str, allow_untrusted_validators: bool, respect_rule_paths: bool=True):
        if False:
            while True:
                i = 10
        self._binary_path = engine_type.get_binary_path()
        self._jobs = jobs or engine_type.default_jobs
        self._engine_type = engine_type
        self._timeout = timeout
        self._max_memory = max_memory
        self._timeout_threshold = timeout_threshold
        self._interfile_timeout = interfile_timeout
        self._optimizations = optimizations
        self._allow_untrusted_validators = allow_untrusted_validators
        self._respect_rule_paths = respect_rule_paths

    def _extract_core_output(self, rules: List[Rule], returncode: int, shell_command: str, core_stdout: str, core_stderr: str) -> Dict[str, Any]:
        if False:
            return 10
        if not core_stderr:
            core_stderr = '<semgrep-core stderr not captured, should be printed above>\n'
        logger.debug(f'--- semgrep-core stderr ---\n{core_stderr}--- end semgrep-core stderr ---')
        if returncode != 0:
            output_json = self._parse_core_output(shell_command, core_stdout, core_stderr, returncode)
            if 'errors' in output_json:
                parsed_output = out.CoreOutput.from_json(output_json)
                errors = parsed_output.errors
                if len(errors) < 1:
                    self._fail('non-zero exit status errors array is empty in json response', shell_command, returncode, core_stdout, core_stderr)
                raise core_error_to_semgrep_error(errors[0])
            else:
                self._fail('non-zero exit status with missing "errors" field in json response', shell_command, returncode, core_stdout, core_stderr)
        output_json = self._parse_core_output(shell_command, core_stdout, core_stderr, returncode)
        return output_json

    def _parse_core_output(self, shell_command: str, semgrep_output: str, semgrep_error_output: str, returncode: int) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        try:
            return cast(Dict[str, Any], json.loads(semgrep_output))
        except ValueError as exn:
            if returncode == -11 or returncode == -9:
                (soft_limit, _hard_limit) = resource.getrlimit(resource.RLIMIT_STACK)
                tip = f"\n                Semgrep exceeded system resources. This may be caused by\n                    1. Stack overflow. Try increasing the stack limit to\n                       `{soft_limit}` by running `ulimit -s {soft_limit}`\n                       before running Semgrep.\n                    2. Out of memory. Try increasing the memory available to\n                       your container (if running in CI). If that is not\n                       possible, run `semgrep` with `--max-memory\n                       $YOUR_MEMORY_LIMIT`.\n                    3. Some extremely niche compiler/c-bindings bug. (We've\n                       never seen this, but it's always possible.)\n                    You can also try reducing the number of processes Semgrep\n                    uses by running `semgrep` with `--jobs 1` (or some other\n                    number of jobs). If you are running in CI, please try\n                    running the same command locally.\n                "
            else:
                tip = f'Semgrep encountered an internal error: {exn}.'
            self._fail(f'{tip}', shell_command, returncode, semgrep_output, semgrep_error_output)
            return {}

    def _fail(self, reason: str, shell_command: str, returncode: int, semgrep_output: str, semgrep_error_output: str) -> None:
        if False:
            print('Hello World!')
        details = with_color(Colors.white, f'semgrep-core exit code: {returncode}\nsemgrep-core command: {shell_command}\nunexpected non-json output while invoking semgrep-core:\n--- semgrep-core stdout ---\n{semgrep_output}--- end semgrep-core stdout ---\n--- semgrep-core stderr ---\n{semgrep_error_output}--- end semgrep-core stderr ---\n')
        raise SemgrepError(f'Error while matching: {reason}\n{details}{PLEASE_FILE_ISSUE_TEXT}')

    @staticmethod
    def plan_core_run(rules: List[Rule], target_manager: TargetManager, *, all_targets: Optional[Set[Path]]=None, product: Optional[out.Product]=None) -> Plan:
        if False:
            return 10
        '\n        Gets the targets to run for each rule\n\n        Returns this information as a list of rule ids and a list of targets with\n        language + index of the rule ids for the rules to run each target on.\n        Semgrep-core will use this to determine what to run (see Input_to_core.atd).\n        Also updates all_targets if set, used by core_runner\n\n        Note: this is a list because a target can appear twice (e.g. Java + Generic)\n        '
        target_info: Dict[Tuple[Path, Language], Tuple[List[int], Set[str]]] = collections.defaultdict(lambda : (list(), set()))
        lockfiles = target_manager.get_all_lockfiles()
        unused_rules = []
        for (rule_num, rule) in enumerate(rules):
            any_target = False
            for language in rule.languages:
                targets = list(target_manager.get_files_for_rule(language, rule.includes, rule.excludes, rule.id, rule.product))
                any_target = any_target or len(targets) > 0
                for target in targets:
                    if all_targets is not None:
                        all_targets.add(target)
                    (rules_nums, products) = target_info[target, language]
                    rules_nums.append(rule_num)
                    products.add(rule.product.to_json_string())
            if not any_target:
                unused_rules.append(rule)
        return Plan([Task(path=target, analyzer=language, products=tuple((out.Product.from_json_string(x) for x in products)), rule_nums=tuple(rule_nums)) for ((target, language), (rule_nums, products)) in target_info.items()], rules, product=product, lockfiles_by_ecosystem=lockfiles, unused_rules=unused_rules)

    def _run_rules_direct_to_semgrep_core_helper(self, rules: List[Rule], target_manager: TargetManager, dump_command_for_core: bool, time_flag: bool, matching_explanations: bool, engine: EngineType, run_secrets: bool, disable_secrets_validation: bool, target_mode_config: TargetModeConfig) -> Tuple[RuleMatchMap, List[SemgrepError], OutputExtra]:
        if False:
            i = 10
            return i + 15
        state = get_state()
        logger.debug(f'Passing whole rules directly to semgrep_core')
        outputs: RuleMatchMap = collections.defaultdict(OrderedRuleMatchList)
        errors: List[SemgrepError] = []
        all_targets: Set[Path] = set()
        file_timeouts: Dict[Path, int] = collections.defaultdict(lambda : 0)
        max_timeout_files: Set[Path] = set()
        parsing_data: ParsingData = ParsingData()
        exit_stack = contextlib.ExitStack()
        rule_file = exit_stack.enter_context((state.env.user_data_folder / 'semgrep_rules.json').open('w+') if dump_command_for_core else tempfile.NamedTemporaryFile('w+', suffix='.json'))
        target_file = exit_stack.enter_context((state.env.user_data_folder / 'semgrep_targets.txt').open('w+') if dump_command_for_core else tempfile.NamedTemporaryFile('w+'))
        if target_mode_config.is_pro_diff_scan:
            diff_target_file = exit_stack.enter_context((state.env.user_data_folder / 'semgrep_diff_targets.txt').open('w+') if dump_command_for_core else tempfile.NamedTemporaryFile('w+'))
        with exit_stack:
            if self._binary_path is None:
                if engine.is_pro:
                    logger.error(f'\nSemgrep Pro is either uninstalled or it is out of date.\n\nTry installing Semgrep Pro (`semgrep install-semgrep-pro`).\n                        ')
                else:
                    logger.error(f'\nCould not find the semgrep-core executable. Your Semgrep install is likely corrupted. Please uninstall Semgrep and try again.\n                        ')
                sys.exit(2)
            cmd = [str(self._binary_path), '-json']
            rule_file_contents = json.dumps({'rules': [rule._raw for rule in rules]}, indent=2, sort_keys=True)
            rule_file.write(rule_file_contents)
            rule_file.flush()
            cmd.extend(['-rules', rule_file.name])
            cmd.extend(['-j', str(self._jobs)])
            if target_mode_config.is_pro_diff_scan:
                diff_targets = target_mode_config.get_diff_targets()
                diff_target_file_contents = '\n'.join([str(path) for path in diff_targets])
                diff_target_file.write(diff_target_file_contents)
                diff_target_file.flush()
                cmd.extend(['-diff_targets', diff_target_file.name])
                cmd.extend(['-diff_depth', str(target_mode_config.get_diff_depth())])
                plan = self.plan_core_run(rules, evolve(target_manager, baseline_handler=None), all_targets=all_targets)
            else:
                plan = self.plan_core_run(rules, target_manager, all_targets=all_targets)
            plan.record_metrics()
            parsing_data.add_targets(plan)
            target_file_contents = json.dumps(plan.to_json())
            target_file.write(target_file_contents)
            target_file.flush()
            cmd.extend(['-targets', target_file.name])
            cmd.extend(['-timeout', str(self._timeout), '-timeout_threshold', str(self._timeout_threshold), '-max_memory', str(self._max_memory)])
            if matching_explanations:
                cmd.append('-matching_explanations')
            if time_flag:
                cmd.append('-json_time')
            if not self._respect_rule_paths:
                cmd.append('-disable_rule_paths')
            vfs_map: Dict[str, bytes] = {target_file.name: target_file_contents.encode('UTF-8'), rule_file.name: rule_file_contents.encode('UTF-8')}
            if self._optimizations != 'none':
                cmd.append('-fast')
            if run_secrets and (not disable_secrets_validation):
                cmd += ['-secrets']
                if not engine.is_pro:
                    raise SemgrepError('Secrets post processors tried to run without the pro-engine.')
            if self._allow_untrusted_validators:
                cmd.append('-allow-untrusted-validators')
            if engine.is_pro:
                if auth.get_token() is None:
                    logger.error('!!!This is a proprietary extension of semgrep.!!!')
                    logger.error('!!!You must be logged in to access this extension!!!')
                elif engine is EngineType.PRO_INTERFILE:
                    logger.error('Semgrep Pro Engine may be slower and show different results than Semgrep OSS.')
                if engine is EngineType.PRO_INTERFILE:
                    targets = target_manager.targets
                    if len(targets) == 1:
                        root = str(targets[0].path)
                    else:
                        raise SemgrepError('Inter-file analysis can only take a single target (for multiple files pass a directory)')
                    cmd += ['-deep_inter_file']
                    cmd += ['-timeout_for_interfile_analysis', str(self._interfile_timeout)]
                    cmd += [root]
                elif engine is EngineType.PRO_INTRAFILE:
                    cmd += ['-deep_intra_file']
            if state.terminal.is_debug:
                cmd += ['--debug']
            show_progress = state.get_cli_ux_flavor() != DesignTreatment.MINIMAL
            total = plan.num_targets * 3 if show_progress else 0
            logger.debug('Running Semgrep engine with command:')
            logger.debug(' '.join(cmd))
            if dump_command_for_core:
                printed_cmd = cmd.copy()
                printed_cmd[0] = str(self._binary_path)
                print(' '.join(printed_cmd))
                sys.exit(0)
            runner = StreamingSemgrepCore(cmd, total=total, engine_type=engine)
            runner.vfs_map = vfs_map
            returncode = runner.execute()
            output_json = self._extract_core_output(rules, returncode, ' '.join(cmd), runner.stdout, runner.stderr)
            core_output = out.CoreOutput.from_json(output_json)
            if core_output.paths.skipped:
                for skip in core_output.paths.skipped:
                    if skip.rule_id:
                        rule_info = f'rule {skip.rule_id}'
                    else:
                        rule_info = 'all rules'
                        logger.verbose(f"skipped '{skip.path}' [{rule_info}]: {skip.reason}: {skip.details}")
            outputs = core_matches_to_rule_matches(rules, core_output)
            parsed_errors = [core_error_to_semgrep_error(e) for e in core_output.errors]
            for err in core_output.errors:
                if isinstance(err.error_type.value, out.Timeout):
                    assert err.location.path is not None
                    file_timeouts[Path(err.location.path.value)] += 1
                    if self._timeout_threshold != 0 and file_timeouts[Path(err.location.path.value)] >= self._timeout_threshold:
                        max_timeout_files.add(Path(err.location.path.value))
                if isinstance(err.error_type.value, (out.LexicalError, out.ParseError, out.PartialParsing, out.OtherParseError, out.AstBuilderError)):
                    parsing_data.add_error(err)
            errors.extend(parsed_errors)
        output_extra = OutputExtra(core_output, all_targets, parsing_data)
        return (outputs, errors, output_extra)

    def _run_rules_direct_to_semgrep_core(self, rules: List[Rule], target_manager: TargetManager, dump_command_for_core: bool, time_flag: bool, matching_explanations: bool, engine: EngineType, run_secrets: bool, disable_secrets_validation: bool, target_mode_config: TargetModeConfig) -> Tuple[RuleMatchMap, List[SemgrepError], OutputExtra]:
        if False:
            while True:
                i = 10
        '\n        Sometimes we may run into synchronicity issues with the latest DeepSemgrep binary.\n        These issues may possibly cause a failure if a user, for instance, updates their\n        version of Semgrep, but does not update to the latest version of DeepSemgrep.\n\n        A short bandaid solution for now is to suggest that a user updates to the latest\n        version, if the DeepSemgrep binary crashes for any reason.\n        '
        try:
            return self._run_rules_direct_to_semgrep_core_helper(rules, target_manager, dump_command_for_core, time_flag, matching_explanations, engine, run_secrets, disable_secrets_validation, target_mode_config)
        except SemgrepError as e:
            raise e
        except Exception as e:
            if engine.is_pro:
                logger.error(f'\n\nSemgrep Pro crashed during execution (unknown reason).\nThis can sometimes happen because either Semgrep Pro or Semgrep is out of date.\n\nTry updating your version of Semgrep Pro (`semgrep install-semgrep-pro`) or your version of Semgrep (`pip install semgrep/brew install semgrep`).\nIf both are up-to-date and the crash persists, please contact support to report an issue!\nWhen reporting the issue, please re-run the semgrep command with the\n`--debug` flag so as to print more details about what happened, if you can.\n\nException raised: `{e}`\n                    ')
                sys.exit(2)
            raise e

    def invoke_semgrep_core(self, target_manager: TargetManager, rules: List[Rule], dump_command_for_core: bool, time_flag: bool, matching_explanations: bool, engine: EngineType, run_secrets: bool, disable_secrets_validation: bool, target_mode_config: TargetModeConfig) -> Tuple[RuleMatchMap, List[SemgrepError], OutputExtra]:
        if False:
            print('Hello World!')
        '\n        Takes in rules and targets and returns object with findings\n        '
        start = datetime.now()
        (findings_by_rule, errors, output_extra) = self._run_rules_direct_to_semgrep_core(rules, target_manager, dump_command_for_core, time_flag, matching_explanations, engine, run_secrets, disable_secrets_validation, target_mode_config)
        logger.debug(f'semgrep ran in {datetime.now() - start} on {len(output_extra.all_targets)} files')
        by_severity = collections.defaultdict(list)
        for (rule, findings) in findings_by_rule.items():
            by_severity[rule.severity.to_json().lower()].extend(findings)
        by_sev_strings = [f'{len(findings)} {sev}' for (sev, findings) in by_severity.items()]
        logger.debug(f"findings summary: {', '.join(by_sev_strings)}")
        return (findings_by_rule, errors, output_extra)

    def validate_configs(self, configs: Tuple[str, ...]) -> Sequence[SemgrepError]:
        if False:
            for i in range(10):
                print('nop')
        if self._binary_path is None:
            raise SemgrepError('semgrep engine not found.')
        metachecks = Config.from_config_list(['p/semgrep-rule-lints'], None)[0].get_rules(True)
        parsed_errors = []
        with tempfile.NamedTemporaryFile('w', suffix='.yaml') as rule_file:
            yaml = YAML()
            yaml.dump({'rules': [metacheck._raw for metacheck in metachecks]}, rule_file)
            rule_file.flush()
            cmd = [str(self._binary_path), '-json', '-check_rules', rule_file.name, *configs]
            show_progress = get_state().get_cli_ux_flavor() != DesignTreatment.MINIMAL
            total = 1 if show_progress else 0
            runner = StreamingSemgrepCore(cmd, total=total, engine_type=self._engine_type)
            returncode = runner.execute()
            output_json = self._extract_core_output(metachecks, returncode, ' '.join(cmd), runner.stdout, runner.stderr)
            core_output = out.CoreOutput.from_json(output_json)
            parsed_errors += [core_error_to_semgrep_error(e) for e in core_output.errors]
        return dedup_errors(parsed_errors)