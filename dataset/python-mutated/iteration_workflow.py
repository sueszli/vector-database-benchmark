import json
from sqlalchemy import Column, Integer, String, Text, Boolean
from superagi.models.base_model import DBBaseModel
from superagi.models.workflows.iteration_workflow_step import IterationWorkflowStep

class IterationWorkflow(DBBaseModel):
    """
    Agent workflows which runs part of each agent execution step

    Attributes:
        id (int): The unique identifier of the agent workflow.
        name (str): The name of the agent workflow.
        description (str): The description of the agent workflow.
    """
    __tablename__ = 'iteration_workflows'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text)
    has_task_queue = Column(Boolean, default=False)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
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
        '\n        Fetches the trigger step ID of the specified iteration workflow.\n\n        Args:\n            session: The session object used for database operations.\n            workflow_id (int): The ID of the agent workflow.\n\n        Returns:\n            int: The ID of the trigger step.\n\n        '
        trigger_step = session.query(IterationWorkflowStep).filter(IterationWorkflowStep.iteration_workflow_id == workflow_id, IterationWorkflowStep.step_type == 'TRIGGER').first()
        return trigger_step

    @classmethod
    def find_workflow_by_name(cls, session, name: str):
        if False:
            print('Hello World!')
        '\n        Finds an IterationWorkflow by name.\n\n        Args:\n            session (Session): SQLAlchemy session object.\n            name (str): Name of the AgentWorkflow.\n\n        Returns:\n            AgentWorkflow: AgentWorkflow object with the given name.\n        '
        return session.query(IterationWorkflow).filter(IterationWorkflow.name == name).first()

    @classmethod
    def find_or_create_by_name(cls, session, name: str, description: str, has_task_queue: bool=False):
        if False:
            print('Hello World!')
        '\n        Finds an IterationWorkflow by name or creates it if it does not exist.\n        Args:\n            session (Session): SQLAlchemy session object.\n            name (str): Name of the AgentWorkflow.\n            description (str): Description of the AgentWorkflow.\n        '
        iteration_workflow = session.query(IterationWorkflow).filter(IterationWorkflow.name == name).first()
        if iteration_workflow is None:
            iteration_workflow = IterationWorkflow(name=name, description=description)
            session.add(iteration_workflow)
            session.commit()
        iteration_workflow.has_task_queue = has_task_queue
        session.commit()
        return iteration_workflow

    @classmethod
    def find_by_id(cls, session, id: int):
        if False:
            print('Hello World!')
        ' Find the workflow step by id'
        return session.query(IterationWorkflow).filter(IterationWorkflow.id == id).first()