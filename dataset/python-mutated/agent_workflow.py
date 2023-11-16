import json
from sqlalchemy import Column, Integer, String, Text
from superagi.models.workflows.agent_workflow_step import AgentWorkflowStep
from superagi.models.base_model import DBBaseModel

class AgentWorkflow(DBBaseModel):
    """
    Agent workflows which runs part of each agent execution step

    Attributes:
        id (int): The unique identifier of the agent workflow.
        name (str): The name of the agent workflow.
        description (str): The description of the agent workflow.
    """
    __tablename__ = 'agent_workflows'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text)

    def __repr__(self):
        if False:
            print('Hello World!')
        '\n        Returns a string representation of the AgentWorkflow object.\n\n        Returns:\n            str: String representation of the AgentWorkflow.\n        '
        return f"AgentWorkflow(id={self.id}, name='{self.name}', description='{self.description}')"

    def to_dict(self):
        if False:
            while True:
                i = 10
        '\n            Converts the AgentWorkflow object to a dictionary.\n\n            Returns:\n                dict: Dictionary representation of the AgentWorkflow.\n        '
        return {'id': self.id, 'name': self.name, 'description': self.description}

    def to_json(self):
        if False:
            while True:
                i = 10
        '\n        Converts the AgentWorkflow object to a JSON string.\n\n        Returns:\n            str: JSON string representation of the AgentWorkflow.\n        '
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates an AgentWorkflow object from a JSON string.\n\n        Args:\n            json_data (str): JSON string representing the AgentWorkflow.\n\n        Returns:\n            AgentWorkflow: AgentWorkflow object created from the JSON string.\n        '
        data = json.loads(json_data)
        return cls(id=data['id'], name=data['name'], description=data['description'])

    @classmethod
    def fetch_trigger_step_id(cls, session, workflow_id):
        if False:
            while True:
                i = 10
        '\n        Fetches the trigger step ID of the specified agent workflow.\n\n        Args:\n            session: The session object used for database operations.\n            workflow_id (int): The ID of the agent workflow.\n\n        Returns:\n            int: The ID of the trigger step.\n\n        '
        trigger_step = session.query(AgentWorkflowStep).filter(AgentWorkflowStep.agent_workflow_id == workflow_id, AgentWorkflowStep.step_type == 'TRIGGER').first()
        return trigger_step

    @classmethod
    def find_by_id(cls, session, id: int):
        if False:
            print('Hello World!')
        'Create or find an agent workflow by name.'
        return session.query(AgentWorkflow).filter(AgentWorkflow.id == id).first()

    @classmethod
    def find_by_name(cls, session, name: str):
        if False:
            print('Hello World!')
        'Create or find an agent workflow by name.'
        return session.query(AgentWorkflow).filter(AgentWorkflow.name == name).first()

    @classmethod
    def find_or_create_by_name(cls, session, name: str, description: str):
        if False:
            return 10
        'Create or find an agent workflow by name.'
        agent_workflow = session.query(AgentWorkflow).filter(AgentWorkflow.name == name).first()
        if agent_workflow is None:
            agent_workflow = AgentWorkflow(name=name, description=description)
            session.add(agent_workflow)
            session.commit()
        return agent_workflow