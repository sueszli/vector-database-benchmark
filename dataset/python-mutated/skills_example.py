import pandas as pd
from pandasai import Agent
from pandasai.llm.openai import OpenAI
from pandasai.skills import skill
employees_data = {'EmployeeID': [1, 2, 3, 4, 5], 'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'], 'Department': ['HR', 'Sales', 'IT', 'Marketing', 'Finance']}
salaries_data = {'EmployeeID': [1, 2, 3, 4, 5], 'Salary': [5000, 6000, 4500, 7000, 5500]}
employees_df = pd.DataFrame(employees_data)
salaries_df = pd.DataFrame(salaries_data)

@skill
def plot_salaries(name: list[str], salary: list[int]) -> str:
    if False:
        print('Hello World!')
    '\n    Displays the bar chart having name on x axis and salaries on y axis using streamlit\n    Args:\n        name (list[str]): Employee name\n        salaries (list[int]): Salaries\n    '
    import matplotlib.pyplot as plt
    plt.bar(name, salary)
    plt.xlabel('Employee Name')
    plt.ylabel('Salary')
    plt.title('Employee Salaries')
    plt.xticks(rotation=45)
llm = OpenAI('YOUR-API-KEY')
agent = Agent([employees_df, salaries_df], config={'llm': llm}, memory_size=10)
agent.add_skills(plot_salaries)
response = agent.chat('Plot the employee salaries against names')
print(response)