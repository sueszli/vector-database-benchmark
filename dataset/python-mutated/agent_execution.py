import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from superagi.models.base_model import DBBaseModel
from superagi.models.workflows.agent_workflow_step import AgentWorkflowStep
from superagi.models.workflows.iteration_workflow import IterationWorkflow

class AgentExecution(DBBaseModel):
    """
    Represents single agent run

    Attributes:
        id (int): The unique identifier of the agent execution.
        status (str): The status of the agent execution. Possible values: 'CREATED', 'RUNNING', 'PAUSED',
            'COMPLETED', 'TERMINATED'.
        name (str): The name of the agent execution.
        agent_id (int): The identifier of the associated agent.
        last_execution_time (datetime): The timestamp of the last execution time.
        num_of_calls (int): The number of calls made during the execution.
        num_of_tokens (int): The number of tokens used during the execution.
        current_agent_step_id (int): The identifier of the current step in the execution.
    """
    __tablename__ = 'agent_executions'
    id = Column(Integer, primary_key=True)
    status = Column(String)
    name = Column(String)
    agent_id = Column(Integer)
    last_execution_time = Column(DateTime)
    num_of_calls = Column(Integer, default=0)
    num_of_tokens = Column(Integer, default=0)
    current_agent_step_id = Column(Integer)
    permission_id = Column(Integer)
    iteration_workflow_step_id = Column(Integer)
    current_feed_group_id = Column(String)
    last_shown_error_id = Column(Integer)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a string representation of the AgentExecution object.\n\n        Returns:\n            str: String representation of the AgentExecution.\n        '
        return f"AgentExecution(id={self.id}, name={self.name}, status='{self.status}', last_execution_time='{self.last_execution_time}', current_agent_step_id={self.current_agent_step_id}, agent_id={self.agent_id}, num_of_calls={self.num_of_calls}, num_of_tokens={self.num_of_tokens},permission_id={self.permission_id}, iteration_workflow_step_id={self.iteration_workflow_step_id})"

    def to_dict(self):
        if False:
            while True:
                i = 10
        '\n        Converts the AgentExecution object to a dictionary.\n\n        Returns:\n            dict: Dictionary representation of the AgentExecution.\n        '
        return {'id': self.id, 'status': self.status, 'name': self.name, 'agent_id': self.agent_id, 'last_execution_time': self.last_execution_time.isoformat(), 'num_of_calls': self.num_of_calls, 'num_of_tokens': self.num_of_tokens, 'current_agent_step_id': self.current_agent_step_id, 'permission_id': self.permission_id, 'iteration_workflow_step_id': self.iteration_workflow_step_id}

    def to_json(self):
        if False:
            while True:
                i = 10
        '\n        Converts the AgentExecution object to a JSON string.\n\n        Returns:\n            str: JSON string representation of the AgentExecution.\n        '
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_data):
        if False:
            print('Hello World!')
        '\n        Creates an AgentExecution object from a JSON string.\n\n        Args:\n            json_data (str): JSON string representing the AgentExecution object.\n\n        Returns:\n            AgentExecution: The created AgentExecution object.\n        '
        data = json.loads(json_data)
        last_execution_time = datetime.fromisoformat(data['last_execution_time'])
        return cls(id=data['id'], status=data['status'], name=data['name'], agent_id=data['agent_id'], last_execution_time=last_execution_time, num_of_calls=data['num_of_calls'], num_of_tokens=data['num_of_tokens'], current_agent_step_id=data['current_agent_step_id'], permission_id=data['permission_id'], iteration_workflow_step_id=data['iteration_workflow_step_id'])

    @classmethod
    def get_agent_execution_from_id(cls, session, agent_execution_id):
        if False:
            return 10
        '\n            Get Agent from agent_id\n\n            Args:\n                session: The database session.\n                agent_execution_id(int) : Unique identifier of an Agent Execution.\n\n            Returns:\n                AgentExecution: AgentExecution object is returned.\n        '
        return session.query(AgentExecution).filter(AgentExecution.id == agent_execution_id).first()

    @classmethod
    def find_by_id(cls, session, execution_id: int):
        if False:
            print('Hello World!')
        '\n        Finds an AgentExecution by its id.\n\n        Args:\n            session: The database session.\n            id (int): The id of the AgentExecution.\n\n        Returns:\n            AgentExecution: The AgentExecution object.\n        '
        return session.query(AgentExecution).filter(AgentExecution.id == execution_id).first()

    @classmethod
    def update_tokens(self, session, agent_execution_id: int, total_tokens: int, new_llm_calls: int=1):
        if False:
            while True:
                i = 10
        agent_execution = session.query(AgentExecution).filter(AgentExecution.id == agent_execution_id).first()
        agent_execution.num_of_calls += new_llm_calls
        agent_execution.num_of_tokens += total_tokens
        session.commit()

    @classmethod
    def assign_next_step_id(cls, session, agent_execution_id: int, next_step_id: int):
        if False:
            while True:
                i = 10
        'Assigns next agent workflow step id to agent execution\n\n        Args:\n            session: The database session.\n            agent_execution_id (int): The id of the agent execution.\n            next_step_id (int): The id of the next agent workflow step.\n        '
        agent_execution = session.query(AgentExecution).filter(AgentExecution.id == agent_execution_id).first()
        agent_execution.current_agent_step_id = next_step_id
        next_step = AgentWorkflowStep.find_by_id(session, next_step_id)
        if next_step.action_type == 'ITERATION_WORKFLOW':
            trigger_step = IterationWorkflow.fetch_trigger_step_id(session, next_step.action_reference_id)
            agent_execution.iteration_workflow_step_id = trigger_step.id
        session.commit()

    @classmethod
    def get_execution_by_agent_id_and_status(cls, session, agent_id: int, status_filter: str):
        if False:
            i = 10
            return i + 15
        db_agent_execution = session.query(AgentExecution).filter(AgentExecution.agent_id == agent_id, AgentExecution.status == status_filter).first()
        return db_agent_execution

    @classmethod
    def get_all_executions_by_status_and_agent_id(cls, session, agent_id, execution_state_change_input, current_status: str):
        if False:
            while True:
                i = 10
        db_execution_arr = []
        if execution_state_change_input.run_ids is not None:
            db_execution_arr = session.query(AgentExecution).filter(AgentExecution.agent_id == agent_id, AgentExecution.status == current_status, AgentExecution.id.in_(execution_state_change_input.run_ids)).all()
        else:
            db_execution_arr = session.query(AgentExecution).filter(AgentExecution.agent_id == agent_id, AgentExecution.status == current_status).all()
        return db_execution_arr

    @classmethod
    def get_all_executions_by_filter_config(cls, session, agent_id: int, filter_config):
        if False:
            for i in range(10):
                print('nop')
        db_execution_query = session.query(AgentExecution).filter(AgentExecution.agent_id == agent_id)
        if filter_config.run_ids is not None:
            db_execution_query = db_execution_query.filter(AgentExecution.id.in_(filter_config.run_ids))
        if filter_config.run_status_filter is not None and filter_config.run_status_filter in ['CREATED', 'RUNNING', 'PAUSED', 'COMPLETED', 'TERMINATED']:
            db_execution_query = db_execution_query.filter(AgentExecution.status == filter_config.run_status_filter)
        db_execution_arr = db_execution_query.all()
        return db_execution_arr

    @classmethod
    def validate_run_ids(cls, session, run_ids: list, organisation_id: int):
        if False:
            i = 10
            return i + 15
        from superagi.models.agent import Agent
        from superagi.models.project import Project
        run_ids = list(set(run_ids))
        agent_ids = session.query(AgentExecution.agent_id).filter(AgentExecution.id.in_(run_ids)).distinct().all()
        agent_ids = [id for (id,) in agent_ids]
        project_ids = session.query(Agent.project_id).filter(Agent.id.in_(agent_ids)).distinct().all()
        project_ids = [id for (id,) in project_ids]
        org_ids = session.query(Project.organisation_id).filter(Project.id.in_(project_ids)).distinct().all()
        org_ids = [id for (id,) in org_ids]
        if len(org_ids) > 1 or org_ids[0] != organisation_id:
            raise Exception(f'one or more run IDs not found')