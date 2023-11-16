import json
from sqlalchemy import Column, Integer, String, DateTime
from superagi.models.base_model import DBBaseModel

class AgentWorkflowStepWait(DBBaseModel):
    """
    Step for a Agent Workflow to wait

    Attributes:
        id (int): The unique identifier of the wait block step.
        name (str): The name of the wait block step.
        description (str): The description of the wait block step.
        delay (int): The delay time in seconds.
        wait_begin_time (DateTime): The start time of the wait block.
    """
    __tablename__ = 'agent_workflow_step_waits'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    unique_id = Column(String)
    delay = Column(Integer)
    wait_begin_time = Column(DateTime)
    status = Column(String)

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Returns a string representation of the WaitBlockStep object.\n\n        Returns:\n            str: String representation of the WaitBlockStep.\n        '
        return f"WaitBlockStep(id={self.id}, name='{self.name}', delay='{self.delay}', wait_begin_time='{self.wait_begin_time}'"

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts the WaitBlockStep object to a dictionary.\n\n        Returns:\n            dict: Dictionary representation of the WaitBlockStep.\n        '
        return {'id': self.id, 'name': self.name, 'delay': self.delay, 'wait_begin_time': self.wait_begin_time}

    def to_json(self):
        if False:
            print('Hello World!')
        '\n        Converts the WaitBlockStep object to a JSON string.\n\n        Returns:\n            str: JSON string representation of the WaitBlockStep.\n        '
        return json.dumps(self.to_dict())

    @classmethod
    def find_by_id(cls, session, step_id: int):
        if False:
            while True:
                i = 10
        return session.query(AgentWorkflowStepWait).filter(AgentWorkflowStepWait.id == step_id).first()

    @classmethod
    def find_or_create_wait(cls, session, step_unique_id: str, description: str, delay: int):
        if False:
            for i in range(10):
                print('nop')
        unique_id = f'{step_unique_id}_wait'
        wait = session.query(AgentWorkflowStepWait).filter(AgentWorkflowStepWait.unique_id == unique_id).first()
        if wait is None:
            wait = AgentWorkflowStepWait(unique_id=unique_id, name=unique_id, delay=delay, description=description, status='PENDING')
            session.add(wait)
        else:
            wait.delay = delay
            wait.description = description
            wait.status = 'PENDING'
        session.commit()
        session.flush()
        return wait