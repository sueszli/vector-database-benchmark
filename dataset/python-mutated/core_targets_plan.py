import collections
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Mapping
from typing import Optional
from typing import Set
from typing import Tuple
from attr import define
from attr import field
from attr import frozen
from boltons.iterutils import get_path
from rich import box
from rich.table import Table
import semgrep.semgrep_interfaces.semgrep_output_v1 as out
from semgrep.rule import Rule
from semgrep.semgrep_interfaces.semgrep_output_v1 import Ecosystem
from semgrep.semgrep_types import Language
from semgrep.state import get_state
from semgrep.verbose_logging import getLogger
logger = getLogger(__name__)

@frozen
class Task:
    path: str = field(converter=str)
    analyzer: Language
    products: Tuple[out.Product, ...]
    rule_nums: Tuple[int, ...]

    @property
    def language_label(self) -> str:
        if False:
            i = 10
            return i + 15
        return '<multilang>' if not self.analyzer.definition.is_target_language else self.analyzer.definition.id

    def to_json(self) -> Any:
        if False:
            while True:
                i = 10
        return {'path': self.path, 'analyzer': self.analyzer, 'products': tuple((x.to_json() for x in self.products))}

class TargetMappings(List[Task]):

    @property
    def rule_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len({rule_num for task in self for rule_num in task.rule_nums})

    @property
    def file_count(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self)

@define
class TaskCounts:
    files: int = 0
    rules: int = 0

class Plan:
    """
    Saves and displays knowledge of what will be run

    to_json: creates the json passed to semgrep_core - see Input_to_core.atd
    log: outputs a summary of how many files will be scanned for each file
    """

    def __init__(self, mappings: List[Task], rules: List[Rule], *, product: Optional[out.Product]=None, lockfiles_by_ecosystem: Optional[Dict[Ecosystem, FrozenSet[Path]]]=None, unused_rules: Optional[List[Rule]]=None):
        if False:
            return 10
        self.target_mappings = TargetMappings(mappings)
        self.rules = rules
        self.product = product
        self.lockfiles_by_ecosystem = lockfiles_by_ecosystem
        self.unused_rules = unused_rules or []

    def split_by_lang_label(self) -> Dict[str, 'TargetMappings']:
        if False:
            while True:
                i = 10
        return self.split_by_lang_label_for_product()

    def split_by_lang_label_for_product(self, product: Optional[out.Product]=None) -> Dict[str, 'TargetMappings']:
        if False:
            for i in range(10):
                print('nop')
        result: Dict[str, TargetMappings] = collections.defaultdict(TargetMappings)
        for task in self.target_mappings:
            result[task.language_label].append(task if product is None else Task(path=task.path, analyzer=task.analyzer, products=(product,), rule_nums=tuple((num for num in task.rule_nums if self.rules[num].product == product))))
        return result

    @lru_cache(maxsize=1000)
    def ecosystems_by_rule_nums(self, rule_nums: Tuple[int]) -> Set[Ecosystem]:
        if False:
            for i in range(10):
                print('nop')
        return {ecosystem for rule_num in rule_nums for ecosystem in self.rules[rule_num].ecosystems}

    def counts_by_ecosystem(self) -> Mapping[Ecosystem, TaskCounts]:
        if False:
            return 10
        result: DefaultDict[Ecosystem, TaskCounts] = collections.defaultdict(TaskCounts)
        for rule in self.rules:
            for ecosystem in rule.ecosystems:
                result[ecosystem].rules += 1
        for task in self.target_mappings:
            for ecosystem in self.ecosystems_by_rule_nums(task.rule_nums):
                result[ecosystem].files += 1
        if self.lockfiles_by_ecosystem is not None:
            unused_ecosystems = {ecosystem for ecosystem in result if not self.lockfiles_by_ecosystem.get(ecosystem)}
            for ecosystem in unused_ecosystems:
                del result[ecosystem]
        return result

    def to_json(self) -> List[Any]:
        if False:
            i = 10
            return i + 15
        return [task.to_json() for task in self.target_mappings]

    @property
    def num_targets(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.target_mappings)

    def rule_count_for_product(self, product: out.Product) -> int:
        if False:
            for i in range(10):
                print('nop')
        rule_nums: Set[int] = set()
        for task in self.target_mappings:
            for rule_num in task.rule_nums:
                if self.rules[rule_num].product == product:
                    rule_nums.add(rule_num)
        return len(rule_nums)

    def table_by_language(self, with_tables_for: Optional[out.Product]=None) -> Table:
        if False:
            for i in range(10):
                print('nop')
        table = Table(box=box.SIMPLE_HEAD, show_edge=False)
        table.add_column('Language')
        table.add_column('Rules', justify='right')
        table.add_column('Files', justify='right')
        plans_by_language = sorted(self.split_by_lang_label_for_product(with_tables_for).items(), key=lambda x: (x[1].file_count, x[1].rule_count), reverse=True)
        for (language, plan) in plans_by_language:
            if plan.rule_count:
                table.add_row(language, str(plan.rule_count), str(plan.file_count))
        return table

    def table_by_ecosystem(self) -> Table:
        if False:
            return 10
        table = Table(box=box.SIMPLE_HEAD, show_edge=False)
        table.add_column('Ecosystem')
        table.add_column('Rules', justify='right')
        table.add_column('Files', justify='right')
        table.add_column('Lockfiles')
        counts_by_ecosystem = self.counts_by_ecosystem()
        for (ecosystem, plan) in sorted(counts_by_ecosystem.items(), key=lambda x: (x[1].files, x[1].rules), reverse=True):
            if self.lockfiles_by_ecosystem is not None:
                lockfile_paths = ', '.join((str(lockfile) for lockfile in self.lockfiles_by_ecosystem.get(ecosystem, [])))
            else:
                lockfile_paths = 'N/A'
            table.add_row(ecosystem.kind, str(plan.rules), str(plan.files), lockfile_paths)
        return table

    def table_by_origin(self, with_tables_for: Optional[out.Product]=None) -> Table:
        if False:
            for i in range(10):
                print('nop')
        table = Table(box=box.SIMPLE_HEAD, show_edge=False)
        table.add_column('Origin')
        table.add_column('Rules', justify='right')
        origin_counts = collections.Counter((get_path(rule.metadata, ('semgrep.dev', 'rule', 'origin'), default='custom') for rule in self.rules if rule.product == with_tables_for))
        for (origin, count) in sorted(origin_counts.items(), key=lambda x: x[1], reverse=True):
            origin_name = origin.replace('_', ' ').capitalize()
            table.add_row(origin_name, str(count))
        return table

    def table_by_sca_analysis(self) -> Table:
        if False:
            return 10
        table = Table(box=box.SIMPLE_HEAD, show_edge=False)
        table.add_column('Analysis')
        table.add_column('Rules', justify='right')
        SCA_ANALYSIS_NAMES = {'reachable': 'Reachability', 'legacy': 'Basic', 'malicious': 'Basic', 'upgrade-only': 'Basic'}
        sca_analysis_counts = collections.Counter((SCA_ANALYSIS_NAMES.get(rule.metadata.get('sca-kind', ''), 'Unknown') for rule in self.rules if isinstance(rule.product.value, out.SCA)))
        for (sca_analysis, count) in sorted(sca_analysis_counts.items(), key=lambda x: x[1], reverse=True):
            sca_analysis_name = sca_analysis.replace('_', ' ').title()
            table.add_row(sca_analysis_name, str(count))
        return table

    def record_metrics(self) -> None:
        if False:
            while True:
                i = 10
        metrics = get_state().metrics
        for language in self.split_by_lang_label():
            metrics.add_feature('language', language)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'<Plan of {len(self.target_mappings)} tasks for {list(self.split_by_lang_label())}>'