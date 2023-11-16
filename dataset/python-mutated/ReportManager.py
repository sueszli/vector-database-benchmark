import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from agbenchmark.reports.processing.graphs import save_single_radar_chart
from agbenchmark.reports.processing.process_report import get_agent_category
from agbenchmark.reports.processing.report_types import Report
from agbenchmark.utils.data_types import AgentBenchmarkConfig
from agbenchmark.utils.utils import get_highest_success_difficulty

class SingletonReportManager:
    instance = None

    def __new__(cls):
        if False:
            return 10
        from agbenchmark.reports.agent_benchmark_config import get_agent_benchmark_config
        if not cls.instance:
            cls.instance = super(SingletonReportManager, cls).__new__(cls)
            agent_benchmark_config = get_agent_benchmark_config()
            benchmark_start_time_dt = datetime.now(timezone.utc)
            cls.REGRESSION_MANAGER = ReportManager(agent_benchmark_config.get_regression_reports_path(), benchmark_start_time_dt)
            cls.INFO_MANAGER = ReportManager(str(agent_benchmark_config.get_reports_path(benchmark_start_time_dt) / 'report.json'), benchmark_start_time_dt)
            cls.INTERNAL_INFO_MANAGER = ReportManager(agent_benchmark_config.get_success_rate_path(), benchmark_start_time_dt)
        return cls.instance

    @classmethod
    def clear_instance(cls):
        if False:
            print('Hello World!')
        cls.instance = None
        cls.REGRESSION_MANAGER = None
        cls.INFO_MANAGER = None
        cls.INTERNAL_INFO_MANAGER = None

class ReportManager:
    """Abstracts interaction with the regression tests file"""

    def __init__(self, filename: str, benchmark_start_time: str):
        if False:
            i = 10
            return i + 15
        self.filename = filename
        self.start_time = time.time()
        self.benchmark_start_time = benchmark_start_time
        self.load()

    def load(self) -> None:
        if False:
            return 10
        if not os.path.exists(self.filename):
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename, 'w') as f:
                pass
        try:
            with open(self.filename, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    data = json.loads(file_content)
                    self.tests = {k: data[k] for k in sorted(data)}
                else:
                    self.tests = {}
        except FileNotFoundError:
            self.tests = {}
        except json.decoder.JSONDecodeError:
            self.tests = {}
        self.save()

    def save(self) -> None:
        if False:
            while True:
                i = 10
        with open(self.filename, 'w') as f:
            json.dump(self.tests, f, indent=4)

    def add_test(self, test_name: str, test_details: dict | list) -> None:
        if False:
            print('Hello World!')
        if test_name.startswith('Test'):
            test_name = test_name[4:]
        self.tests[test_name] = test_details
        self.save()

    def remove_test(self, test_name: str) -> None:
        if False:
            print('Hello World!')
        if test_name in self.tests:
            del self.tests[test_name]
            self.save()

    def reset(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.tests = {}
        self.save()

    def end_info_report(self, config: AgentBenchmarkConfig) -> None:
        if False:
            return 10
        command = ' '.join(sys.argv)
        self.tests = {'command': command.split(os.sep)[-1], 'benchmark_git_commit_sha': '---', 'agent_git_commit_sha': '---', 'completion_time': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00'), 'benchmark_start_time': self.benchmark_start_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'), 'metrics': {'run_time': str(round(time.time() - self.start_time, 2)) + ' seconds', 'highest_difficulty': get_highest_success_difficulty(self.tests), 'total_cost': self.get_total_costs()}, 'tests': copy.copy(self.tests), 'config': {k: v for (k, v) in json.loads(config.json()).items() if v is not None}}
        Report.parse_obj(self.tests)
        converted_data = Report.parse_obj(self.tests)
        agent_categories = get_agent_category(converted_data)
        if len(agent_categories) > 1:
            save_single_radar_chart(agent_categories, config.get_reports_path(self.benchmark_start_time) / 'radar_chart.png')
        self.save()

    def get_total_costs(self):
        if False:
            while True:
                i = 10
        total_cost = 0
        all_costs_none = True
        for (test_name, test_data) in self.tests.items():
            cost = test_data['metrics'].get('cost', 0)
            if cost is not None:
                all_costs_none = False
                total_cost += cost
        if all_costs_none:
            total_cost = None
        return total_cost