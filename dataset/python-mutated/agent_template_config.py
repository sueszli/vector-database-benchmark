import json
from sqlalchemy import Column, Integer, String, Text
from superagi.models.base_model import DBBaseModel

class AgentTemplateConfig(DBBaseModel):
    """
    Agent template related configurations like goals, instructions, constraints and tools are stored here

    Attributes:
        id (int): The unique identifier of the agent template config.
        agent_template_id (int): The identifier of the associated agent template.
        key (str): The key of the configuration setting.
        value (str): The value of the configuration setting.
    """
    __tablename__ = 'agent_template_configs'
    id = Column(Integer, primary_key=True)
    agent_template_id = Column(Integer)
    key = Column(String)
    value = Column(Text)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a string representation of the AgentTemplateConfig object.\n\n        Returns:\n            str: String representation of the AgentTemplateConfig.\n        '
        return f"AgentTemplateConfig(id={self.id}, agent_template_id='{self.agent_template_id}', key='{self.key}', value='{self.value}')"

    def to_dict(self):
        if False:
            return 10
        '\n        Converts the AgentTemplateConfig object to a dictionary.\n\n        Returns:\n            dict: Dictionary representation of the AgentTemplateConfig.\n        '
        return {'id': self.id, 'agent_template_id': self.agent_template_id, 'key': self.key, 'value': self.value}

    def to_json(self):
        if False:
            while True:
                i = 10
        '\n        Converts the AgentTemplateConfig object to a JSON string.\n\n        Returns:\n            str: JSON string representation of the AgentTemplateConfig.\n        '
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_data):
        if False:
            while True:
                i = 10
        '\n        Creates an AgentTemplateConfig object from a JSON string.\n\n        Args:\n            json_data (str): JSON string representing the AgentTemplateConfig.\n\n        Returns:\n            AgentTemplateConfig: AgentTemplateConfig object created from the JSON string.\n        '
        data = json.loads(json_data)
        return cls(id=data['id'], agent_template_id=data['agent_template_id'], key=data['key'], value=data['value'])