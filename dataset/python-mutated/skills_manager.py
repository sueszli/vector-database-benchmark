from typing import List

class SkillsManager:
    """
    Manages Custom added Skills and tracks used skills for the query
    """
    _skills: List
    _used_skills: List[str]

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._skills = []
        self._used_skills = []

    def add_skills(self, *skills):
        if False:
            return 10
        '\n        Add skills to the list of skills. If a skill with the same name\n             already exists, raise an error.\n\n        Args:\n            *skills: Variable number of skill objects to add.\n        '
        for skill in skills:
            if any((existing_skill.name == skill.name for existing_skill in self._skills)):
                raise ValueError(f"Skill with name '{skill.name}' already exists.")
        self._skills.extend(skills)

    def skill_exists(self, name: str):
        if False:
            return 10
        '\n        Check if a skill with the given name exists in the list of skills.\n\n        Args:\n            name (str): The name of the skill to check.\n\n        Returns:\n            bool: True if a skill with the given name exists, False otherwise.\n        '
        return any((skill.name == name for skill in self._skills))

    def get_skill_by_func_name(self, name: str):
        if False:
            print('Hello World!')
        '\n        Get a skill by its name.\n\n        Args:\n            name (str): The name of the skill to retrieve.\n\n        Returns:\n            Skill or None: The skill with the given name, or None if not found.\n        '
        return next((skill for skill in self._skills if skill.name == name), None)

    def add_used_skill(self, skill: str):
        if False:
            return 10
        if self.skill_exists(skill):
            self._used_skills.append(skill)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Present all skills\n        Returns:\n            str: _description_\n        '
        skills_repr = ''
        for skill in self._skills:
            skills_repr = skills_repr + skill.print
        return skills_repr

    def prompt_display(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Displays skills for prompt\n        '
        if len(self._skills) == 0:
            return
        return '\nYou can also use the following functions, if relevant:\n\n' + self.__str__()

    @property
    def used_skills(self):
        if False:
            for i in range(10):
                print('nop')
        return self._used_skills

    @used_skills.setter
    def used_skills(self, value):
        if False:
            while True:
                i = 10
        self._used_skills = value

    @property
    def skills(self):
        if False:
            return 10
        return self._skills