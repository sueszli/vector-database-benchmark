from __future__ import annotations
import abc
from collections import defaultdict
from dataclasses import dataclass
import itertools
import os
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple
from jinja2 import Environment
from jinja2 import FileSystemLoader
import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import mannwhitneyu
Moments = Tuple[float, float]
Samples = Dict[str, List[float]]

class BaseMetric(object, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Metric name displayed in final report.'
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def precision(self) -> int:
        if False:
            while True:
                i = 10
        'Number of digits following decimal point displayed in final report.'
        raise NotImplementedError

    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame) -> list[float]:
        if False:
            for i in range(10):
                print('nop')
        'Calculates metric for each study in data frame.'
        raise NotImplementedError

class BestValueMetric(BaseMetric):
    name = 'Best value'
    precision = 6

    def calculate(self, data: pd.DataFrame) -> list[float]:
        if False:
            i = 10
            return i + 15
        return data.groupby('uuid').generalization.min().values

class AUCMetric(BaseMetric):
    name = 'AUC'
    precision = 3

    def calculate(self, data: pd.DataFrame) -> list[float]:
        if False:
            return 10
        aucs: list[float] = list()
        for (_, grp) in data.groupby('uuid'):
            auc = np.sum(grp.generalization.cummin())
            aucs.append(auc / grp.shape[0])
        return aucs

class ElapsedMetric(BaseMetric):
    name = 'Elapsed'
    precision = 3

    def calculate(self, data: pd.DataFrame) -> list[float]:
        if False:
            while True:
                i = 10
        time_cols = ['suggest', 'observe']
        return data.groupby('uuid')[time_cols].sum().sum(axis=1).values

class PartialReport:

    def __init__(self, data: pd.DataFrame) -> None:
        if False:
            print('Hello World!')
        self._data = data

    @property
    def optimizers(self) -> list[str]:
        if False:
            while True:
                i = 10
        return list(self._data.opt.unique())

    @classmethod
    def from_json(cls, path: str) -> 'PartialReport':
        if False:
            while True:
                i = 10
        data = pd.read_json(path)
        return cls(data)

    def summarize_solver(self, solver: str, metric: BaseMetric) -> Moments:
        if False:
            i = 10
            return i + 15
        solver_data = self._data[self._data.opt == solver]
        if solver_data.shape[0] == 0:
            raise ValueError(f'{solver} not found in report.')
        run_metrics = metric.calculate(solver_data)
        return (np.mean(run_metrics).item(), np.var(run_metrics).item())

    def sample_performance(self, metric: BaseMetric) -> Samples:
        if False:
            return 10
        performance: dict[str, list[float]] = {}
        for (solver, data) in self._data.groupby('opt'):
            run_metrics = metric.calculate(data)
            performance[solver] = run_metrics
        return performance

class DewanckerRanker:

    def __init__(self, metrics: list[BaseMetric]) -> None:
        if False:
            return 10
        self._metrics = metrics
        self._ranking: list[str] | None = None
        self._borda: np.ndarray | None = None

    def __iter__(self) -> Generator[tuple[str, int], None, None]:
        if False:
            for i in range(10):
                print('nop')
        yield from zip(self.solvers, self.borda)

    @property
    def solvers(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        if self._ranking is None:
            raise ValueError('Call rank first.')
        return self._ranking

    @property
    def borda(self) -> np.ndarray:
        if False:
            return 10
        if self._borda is None:
            raise ValueError('Call rank first.')
        return self._borda

    @staticmethod
    def pick_alpha(report: PartialReport) -> float:
        if False:
            i = 10
            return i + 15
        num_optimizers = len(report.optimizers)
        candidates = [0.075, 0.05, 0.025, 0.01] * 4 / np.repeat([1, 10, 100, 1000], 4)
        for cand in candidates:
            if 1 - np.power(1 - cand, binom(num_optimizers, 2)) < 0.05:
                return cand
        return candidates[-1]

    def _set_ranking(self, wins: dict[str, int]) -> None:
        if False:
            print('Hello World!')
        sorted_wins = [k for (k, _) in sorted(wins.items(), key=lambda x: x[1])]
        self._ranking = sorted_wins[::-1]

    def _set_borda(self, wins: dict[str, int]) -> None:
        if False:
            i = 10
            return i + 15
        sorted_wins = np.array(sorted(wins.values()))
        (num_wins, num_ties) = np.unique(sorted_wins, return_counts=True)
        points = np.searchsorted(sorted_wins, num_wins)
        self._borda = np.repeat(points, num_ties)[::-1]

    def rank(self, report: PartialReport) -> None:
        if False:
            i = 10
            return i + 15
        wins: dict[str, int] = defaultdict(int)
        alpha = DewanckerRanker.pick_alpha(report)
        for metric in self._metrics:
            samples = report.sample_performance(metric)
            for (optim_a, optim_b) in itertools.permutations(samples, 2):
                (_, p_val) = mannwhitneyu(samples[optim_a], samples[optim_b], alternative='less')
                if p_val < alpha:
                    wins[optim_a] += 1
            all_wins = [wins[optimizer] for optimizer in report.optimizers]
            no_ties = len(all_wins) == len(np.unique(all_wins))
            if no_ties:
                break
        wins = {optimzier: wins[optimzier] for optimzier in report.optimizers}
        self._set_ranking(wins)
        self._set_borda(wins)

@dataclass
class Solver:
    rank: int
    name: str
    results: list[str]

@dataclass
class Problem:
    number: int
    name: str
    metrics: list[BaseMetric]
    solvers: list[Solver]

class BayesmarkReportBuilder:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.solvers: set[str] = set()
        self.datasets: set[str] = set()
        self.models: set[str] = set()
        self.firsts: dict[str, int] = defaultdict(int)
        self.borda: dict[str, int] = defaultdict(int)
        self.metric_precedence = ''
        self.problems: list[Problem] = []

    def set_precedence(self, metrics: list[BaseMetric]) -> None:
        if False:
            print('Hello World!')
        self.metric_precedence = ' -> '.join([m.name for m in metrics])

    def add_problem(self, name: str, report: PartialReport, ranking: DewanckerRanker, metrics: list[BaseMetric]) -> 'BayesmarkReportBuilder':
        if False:
            while True:
                i = 10
        solvers: list[Solver] = list()
        positions = np.abs(ranking.borda - (max(ranking.borda) + 1))
        for (pos, solver) in zip(positions, ranking.solvers):
            self.solvers.add(solver)
            results: list[str] = list()
            for metric in metrics:
                (mean, variance) = report.summarize_solver(solver, metric)
                precision = metric.precision
                results.append(f'{mean:.{precision}f} +- {np.sqrt(variance):.{precision}f}')
            solvers.append(Solver(pos, solver, results))
        problem_number = len(self.problems) + 1
        self.problems.append(Problem(problem_number, name, metrics, solvers))
        return self

    def update_leaderboard(self, ranking: DewanckerRanker) -> 'BayesmarkReportBuilder':
        if False:
            return 10
        for (solver, borda) in ranking:
            if borda == max(ranking.borda):
                self.firsts[solver] += 1
            self.borda[solver] += borda
        return self

    def add_dataset(self, dataset: str) -> 'BayesmarkReportBuilder':
        if False:
            for i in range(10):
                print('nop')
        self.datasets.update(dataset)
        return self

    def add_model(self, model: str) -> 'BayesmarkReportBuilder':
        if False:
            return 10
        self.models.update(model)
        return self

    def assemble_report(self) -> str:
        if False:
            print('Hello World!')
        loader = FileSystemLoader(os.path.join('benchmarks', 'bayesmark'))
        env = Environment(loader=loader)
        report_template = env.get_template('report_template.md')
        return report_template.render(report=self)

def build_report() -> None:
    if False:
        return 10
    metrics = [BestValueMetric(), AUCMetric()]
    report_builder = BayesmarkReportBuilder()
    report_builder.set_precedence(metrics)
    for partial_name in os.listdir('partial'):
        (dataset, model, *_) = partial_name.split('-')
        problem_name = f'{dataset.capitalize()}-{model}'
        path = os.path.join('partial', partial_name)
        partial = PartialReport.from_json(path)
        ranking = DewanckerRanker(metrics)
        ranking.rank(partial)
        elapsed = ElapsedMetric()
        all_metrics = [*metrics, elapsed]
        report_builder = report_builder.add_problem(problem_name, partial, ranking, all_metrics).add_dataset(dataset).add_model(model).update_leaderboard(ranking)
    report = report_builder.assemble_report()
    with open(os.path.join('report', 'benchmark-report.md'), 'w') as file:
        file.write(report)
if __name__ == '__main__':
    os.makedirs('report', exist_ok=True)
    build_report()