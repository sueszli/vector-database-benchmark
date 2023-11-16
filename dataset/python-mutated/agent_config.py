from fastapi import HTTPException
from sqlalchemy import Column, Integer, Text, String
from typing import Union
from superagi.config.config import get_config
from superagi.helper.encyption_helper import decrypt_data
from superagi.models.base_model import DBBaseModel
from superagi.models.configuration import Configuration
from superagi.models.models_config import ModelsConfig
from superagi.types.model_source_types import ModelSourceType
from superagi.models.tool import Tool
from superagi.controllers.types.agent_execution_config import AgentRunIn

class AgentConfiguration(DBBaseModel):
    """
    Agent related configurations like goals, instructions, constraints and tools are stored here

    Attributes:
        id (int): The unique identifier of the agent configuration.
        agent_id (int): The identifier of the associated agent.
        key (str): The key of the configuration setting.
        value (str): The value of the configuration setting.
    """
    __tablename__ = 'agent_configurations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(Integer)
    key = Column(String)
    value = Column(Text)

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Returns a string representation of the Agent Configuration object.\n\n        Returns:\n            str: String representation of the Agent Configuration.\n\n        '
        return f'AgentConfiguration(id={self.id}, key={self.key}, value={self.value})'

    @classmethod
    def update_agent_configurations_table(cls, session, agent_id: Union[int, None], updated_details: AgentRunIn):
        if False:
            for i in range(10):
                print('nop')
        updated_details_dict = updated_details.dict()
        agent_toolkits_config = session.query(AgentConfiguration).filter(AgentConfiguration.agent_id == agent_id, AgentConfiguration.key == 'toolkits').first()
        if agent_toolkits_config:
            agent_toolkits_config.value = str(updated_details_dict['toolkits'])
        else:
            agent_toolkits_config = AgentConfiguration(agent_id=agent_id, key='toolkits', value=str(updated_details_dict['toolkits']))
            session.add(agent_toolkits_config)
        knowledge_config = session.query(AgentConfiguration).filter(AgentConfiguration.agent_id == agent_id, AgentConfiguration.key == 'knowledge').first()
        if knowledge_config:
            knowledge_config.value = str(updated_details_dict['knowledge'])
        else:
            knowledge_config = AgentConfiguration(agent_id=agent_id, key='knowledge', value=str(updated_details_dict['knowledge']))
            session.add(knowledge_config)
        agent_configs = session.query(AgentConfiguration).filter(AgentConfiguration.agent_id == agent_id).all()
        for agent_config in agent_configs:
            if agent_config.key in updated_details_dict:
                agent_config.value = str(updated_details_dict[agent_config.key])
        session.commit()
        return 'Details updated successfully'

    @classmethod
    def get_model_api_key(cls, session, agent_id: int, model: str):
        if False:
            i = 10
            return i + 15
        '\n        Get the model API key from the agent id.\n\n        Args:\n            session (Session): The database session\n            agent_id (int): The agent identifier\n            model (str): The model name\n\n        Returns:\n            str: The model API key.\n        '
        config_model = ModelsConfig.fetch_value_by_agent_id(session, agent_id, model)
        return config_model

    @classmethod
    def get_agent_config_by_key_and_agent_id(cls, session, key: str, agent_id: int):
        if False:
            i = 10
            return i + 15
        agent_config = session.query(AgentConfiguration).filter(AgentConfiguration.agent_id == agent_id, AgentConfiguration.key == key).first()
        return agent_config