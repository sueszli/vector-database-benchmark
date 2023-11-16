from typing import Optional
from unittest.mock import MagicMock, Mock, patch
import uuid
import pandas as pd
import pytest
from pandasai.agent import Agent
from pandasai.helpers.code_manager import CodeExecutionContext, CodeManager
from pandasai.helpers.skills_manager import SkillsManager
from pandasai.llm.fake import FakeLLM
from pandasai.skills import skill
from pandasai.smart_dataframe import SmartDataframe

class TestSkills:

    @pytest.fixture
    def llm(self, output: Optional[str]=None):
        if False:
            return 10
        return FakeLLM(output=output)

    @pytest.fixture
    def sample_df(self):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame({'country': ['United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Canada', 'Australia', 'Japan', 'China'], 'gdp': [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064], 'happiness_index': [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]})

    @pytest.fixture
    def smart_dataframe(self, llm, sample_df):
        if False:
            print('Hello World!')
        return SmartDataframe(sample_df, config={'llm': llm, 'enable_cache': False})

    @pytest.fixture
    def code_manager(self, smart_dataframe: SmartDataframe):
        if False:
            return 10
        return smart_dataframe.lake._code_manager

    @pytest.fixture
    def exec_context(self) -> MagicMock:
        if False:
            print('Hello World!')
        context = MagicMock(spec=CodeExecutionContext)
        return context

    @pytest.fixture
    def agent(self, llm, sample_df):
        if False:
            return 10
        return Agent(sample_df, config={'llm': llm, 'enable_cache': False})

    def test_add_skills(self):
        if False:
            print('Hello World!')
        skills_manager = SkillsManager()
        skill1 = Mock(name='SkillA', print='SkillA Print')
        skill2 = Mock(name='SkillB', print='SkillB Print')
        skills_manager.add_skills(skill1, skill2)
        assert skill1 in skills_manager.skills
        assert skill2 in skills_manager.skills
        try:
            skills_manager.add_skills(skill1)
        except ValueError as e:
            assert str(e) == f"Skill with name '{skill1.name}' already exists."
        else:
            assert False, 'Expected ValueError'

    def test_skill_exists(self):
        if False:
            for i in range(10):
                print('nop')
        skills_manager = SkillsManager()
        skill1 = MagicMock()
        skill2 = MagicMock()
        skill1.name = 'SkillA'
        skill2.name = 'SkillB'
        skills_manager.add_skills(skill1, skill2)
        assert skills_manager.skill_exists('SkillA')
        assert skills_manager.skill_exists('SkillB')
        assert not skills_manager.skill_exists('SkillC')

    def test_get_skill_by_func_name(self):
        if False:
            for i in range(10):
                print('nop')
        skills_manager = SkillsManager()
        skill1 = Mock()
        skill2 = Mock()
        skill1.name = 'SkillA'
        skill2.name = 'SkillB'
        skills_manager.add_skills(skill1, skill2)
        retrieved_skill = skills_manager.get_skill_by_func_name('SkillA')
        assert retrieved_skill == skill1
        retrieved_skill = skills_manager.get_skill_by_func_name('SkillC')
        assert retrieved_skill is None

    def test_add_used_skill(self):
        if False:
            for i in range(10):
                print('nop')
        skills_manager = SkillsManager()
        skill1 = Mock()
        skill2 = Mock()
        skill1.name = 'SkillA'
        skill2.name = 'SkillB'
        skills_manager.add_skills(skill1, skill2)
        skills_manager.add_used_skill('SkillA')
        skills_manager.add_used_skill('SkillB')
        assert 'SkillA' in skills_manager.used_skills
        assert 'SkillB' in skills_manager.used_skills

    def test_prompt_display(self):
        if False:
            for i in range(10):
                print('nop')
        skills_manager = SkillsManager()
        skill1 = Mock()
        skill2 = Mock()
        skill1.name = 'SkillA'
        skill2.name = 'SkillB'
        skill1.print = 'SkillA'
        skill2.print = 'SkillB'
        skills_manager.add_skills(skill1, skill2)
        prompt = skills_manager.prompt_display()
        assert 'You can also use the following functions' in prompt
        skills_manager._skills = []
        prompt = skills_manager.prompt_display()
        assert prompt is None

    @patch('pandasai.skills.inspect.signature', return_value='(a, b, c)')
    def test_skill_decorator(self, mock_inspect_signature):
        if False:
            print('Hello World!')

        @skill
        def skill_a(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return 'SkillA Result'

        @skill
        def skill_b(*args, **kwargs):
            if False:
                return 10
            return 'SkillB Result'
        assert skill_a() == 'SkillA Result'
        assert skill_b() == 'SkillB Result'
        assert skill_a.name == 'skill_a'
        assert skill_b.name == 'skill_b'
        assert skill_a.func_def == 'def pandasai.skills.skill_a(a, b, c)'
        assert skill_b.func_def == 'def pandasai.skills.skill_b(a, b, c)'
        assert skill_a.print == '\n<function>\ndef pandasai.skills.skill_a(a, b, c)\n\n</function>\n'
        assert skill_b.print == '\n<function>\ndef pandasai.skills.skill_b(a, b, c)\n\n</function>\n'

    @patch('pandasai.skills.inspect.signature', return_value='(a, b, c)')
    def test_skill_decorator_test_codc(self, llm):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'country': []})
        df = SmartDataframe(df, config={'llm': llm, 'enable_cache': False})

        @skill
        def plot_salaries(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test skill A\n            Args:\n                arg(str)\n            '
            return 'SkillA Result'
        function_def = '\n            Test skill A\n            Args:\n                arg(str)\n'
        assert function_def in plot_salaries.print

    def test_add_skills_with_agent(self, agent: Agent):
        if False:
            i = 10
            return i + 15

        @skill
        def skill_a(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return 'SkillA Result'

        @skill
        def skill_b(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return 'SkillB Result'
        agent.add_skills(skill_a)
        assert len(agent._lake._skills.skills) == 1
        agent._lake._skills._skills = []
        agent.add_skills(skill_a, skill_b)
        assert len(agent._lake._skills.skills) == 2

    def test_add_skills_with_smartDataframe(self, smart_dataframe: SmartDataframe):
        if False:
            return 10

        @skill
        def skill_a(*args, **kwargs):
            if False:
                print('Hello World!')
            return 'SkillA Result'

        @skill
        def skill_b(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return 'SkillB Result'
        smart_dataframe.add_skills(skill_a)
        assert len(smart_dataframe._lake._skills.skills) == 1
        smart_dataframe._lake._skills._skills = []
        smart_dataframe.add_skills(skill_a, skill_b)
        assert len(smart_dataframe._lake._skills.skills) == 2

    def test_run_prompt(self, llm):
        if False:
            return 10
        df = pd.DataFrame({'country': []})
        df = SmartDataframe(df, config={'llm': llm, 'enable_cache': False})
        function_def = '\n<function>\ndef pandasai.skills.plot_salaries(merged_df: pandas.core.frame.DataFrame) -> str\n\n</function>\n'

        @skill
        def plot_salaries(merged_df: pd.DataFrame) -> str:
            if False:
                return 10
            import matplotlib.pyplot as plt
            plt.bar(merged_df['Name'], merged_df['Salary'])
            plt.xlabel('Employee Name')
            plt.ylabel('Salary')
            plt.title('Employee Salaries')
            plt.xticks(rotation=45)
            plt.savefig('temp_chart.png')
            plt.close()
        df.add_skills(plot_salaries)
        df.chat('How many countries are in the dataframe?')
        last_prompt = df.last_prompt
        assert function_def in last_prompt

    def test_run_prompt_agent(self, agent):
        if False:
            print('Hello World!')
        function_def = '\n<function>\ndef pandasai.skills.plot_salaries(merged_df: pandas.core.frame.DataFrame) -> str\n\n</function>\n'

        @skill
        def plot_salaries(merged_df: pd.DataFrame) -> str:
            if False:
                print('Hello World!')
            import matplotlib.pyplot as plt
            plt.bar(merged_df['Name'], merged_df['Salary'])
            plt.xlabel('Employee Name')
            plt.ylabel('Salary')
            plt.title('Employee Salaries')
            plt.xticks(rotation=45)
            plt.savefig('temp_chart.png')
            plt.close()
        agent.add_skills(plot_salaries)
        agent.chat('How many countries are in the dataframe?')
        last_prompt = agent._lake.last_prompt
        assert function_def in last_prompt

    def test_run_prompt_without_skills(self, agent):
        if False:
            i = 10
            return i + 15
        agent.chat('How many countries are in the dataframe?')
        last_prompt = agent._lake.last_prompt
        assert '<function>' not in last_prompt
        assert '</function>' not in last_prompt
        assert 'You can also use the following functions, if relevant:' not in last_prompt

    def test_code_exec_with_skills_no_use(self, code_manager: CodeManager, exec_context: MagicMock):
        if False:
            return 10
        code = "def analyze_data(dfs):\n    return {'type': 'number', 'value': 1 + 1}"
        skill1 = MagicMock()
        skill1.name = 'SkillA'
        exec_context._skills_manager._skills = [skill1]
        code_manager.execute_code(code, exec_context)
        assert len(exec_context._skills_manager.used_skills) == 0

    def test_code_exec_with_skills(self, code_manager: CodeManager):
        if False:
            while True:
                i = 10
        code = "def analyze_data(dfs):\n    plot_salaries()\n    return {'type': 'number', 'value': 1 + 1}"

        @skill
        def plot_salaries() -> str:
            if False:
                return 10
            return 'plot_salaries'
        code_manager._middlewares = []
        sm = SkillsManager()
        sm.add_skills(plot_salaries)
        exec_context = CodeExecutionContext(uuid.uuid4(), sm)
        code_manager.execute_code(code, exec_context)
        assert len(exec_context._skills_manager.used_skills) == 1
        assert exec_context._skills_manager.used_skills[0] == 'plot_salaries'