import json
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from superagi.lib.logger import logger
from superagi.models.base_model import DBBaseModel
from superagi.models.workflows.agent_workflow_step_tool import AgentWorkflowStepTool
from superagi.models.workflows.agent_workflow_step_wait import AgentWorkflowStepWait
from superagi.models.workflows.iteration_workflow import IterationWorkflow

class AgentWorkflowStep(DBBaseModel):
    """
    Step of an agent workflow

    Attributes:
        id (int): The unique identifier of the agent workflow step.
        agent_workflow_id (int): The ID of the agent workflow to which this step belongs.
        unique_id (str): The unique identifier of the step.
        step_type (str): The type of the step (TRIGGER, NORMAL).
        action_type (str): The type of the action (TOOL, ITERATION_WORKFLOW, LLM).
        action_reference_id: Reference id of the tool/iteration workflow/llm
        next_steps: Next steps output and step id.
    """
    __tablename__ = 'agent_workflow_steps'
    id = Column(Integer, primary_key=True)
    agent_workflow_id = Column(Integer)
    unique_id = Column(String)
    step_type = Column(String)
    action_type = Column(String)
    action_reference_id = Column(Integer)
    next_steps = Column(JSONB)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a string representation of the AgentWorkflowStep object.\n\n        Returns:\n            str: String representation of the AgentWorkflowStep.\n        '
        return f"AgentWorkflowStep(id={self.id}, status='{self.agent_workflow_id}', prompt='{self.unique_id}', agent_id={self.step_type}, action_type={self.action_type}, action_reference_id={self.action_reference_id}, next_steps={self.next_steps})"

    def to_dict(self):
        if False:
            print('Hello World!')
        '\n        Converts the AgentWorkflowStep object to a dictionary.\n\n        Returns:\n            dict: Dictionary representation of the AgentWorkflowStep.\n        '
        return {'id': self.id, 'agent_workflow_id': self.agent_workflow_id, 'unique_id': self.unique_id, 'step_type': self.step_type, 'next_steps': self.next_steps, 'action_type': self.action_type, 'action_reference_id': self.action_reference_id}

    def to_json(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts the AgentWorkflowStep object to a JSON string.\n\n        Returns:\n            str: JSON string representation of the AgentWorkflowStep.\n        '
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_data):
        if False:
            return 10
        '\n        Creates an AgentWorkflowStep object from a JSON string.\n\n        Args:\n            json_data (str): JSON string representing the AgentWorkflowStep.\n\n        Returns:\n            AgentWorkflowStep: AgentWorkflowStep object created from the JSON string.\n        '
        data = json.loads(json_data)
        return cls(id=data['id'], agent_workflow_id=data['agent_workflow_id'], unique_id=data['unique_id'], step_type=data['step_type'], action_type=data['action_type'], action_reference_id=data['action_reference_id'], next_steps=data['next_steps'])

    @classmethod
    def find_by_unique_id(cls, session, unique_id: str):
        if False:
            i = 10
            return i + 15
        ' Adds a workflows step in the next_steps column'
        return session.query(AgentWorkflowStep).filter(AgentWorkflowStep.unique_id == unique_id).first()

    @classmethod
    def find_by_id(cls, session, step_id: int):
        if False:
            return 10
        ' Find the workflow step by id'
        return session.query(AgentWorkflowStep).filter(AgentWorkflowStep.id == step_id).first()

    @classmethod
    def find_or_create_tool_workflow_step(cls, session, agent_workflow_id: int, unique_id: str, tool_name: str, input_instruction: str, output_instruction: str='', step_type='NORMAL', history_enabled: bool=True, completion_prompt: str=None):
        if False:
            i = 10
            return i + 15
        ' Find or create a tool workflow step\n\n        Args:\n            session: db session\n            agent_workflow_id: id of the agent workflow\n            unique_id: unique id of the step\n            tool_name: name of the tool\n            input_instruction: input instruction of the tool\n            output_instruction: output instruction of the tool\n            step_type: type of the step\n            history_enabled: whether to enable history for the step\n            completion_prompt: completion prompt in the llm\n\n        Returns:\n            AgentWorkflowStep.\n        '
        workflow_step = session.query(AgentWorkflowStep).filter(AgentWorkflowStep.agent_workflow_id == agent_workflow_id, AgentWorkflowStep.unique_id == unique_id).first()
        if completion_prompt is None:
            completion_prompt = f'Respond with only valid JSON conforming to the given json schema. Response should contain tool name and tool arguments to achieve the given instruction.'
        step_tool = AgentWorkflowStepTool.find_or_create_tool(session, unique_id, tool_name, input_instruction, output_instruction, history_enabled, completion_prompt)
        if workflow_step is None:
            workflow_step = AgentWorkflowStep(unique_id=unique_id, step_type=step_type, agent_workflow_id=agent_workflow_id)
            session.add(workflow_step)
            session.commit()
        workflow_step.step_type = step_type
        workflow_step.agent_workflow_id = agent_workflow_id
        workflow_step.action_reference_id = step_tool.id
        workflow_step.action_type = 'TOOL'
        workflow_step.next_steps = []
        workflow_step.completion_prompt = completion_prompt
        session.commit()
        return workflow_step

    @classmethod
    def find_or_create_wait_workflow_step(cls, session, agent_workflow_id: int, unique_id: str, wait_description: str, delay: int, step_type='NORMAL'):
        if False:
            return 10
        ' Find or create a wait workflow step'
        logger.info('Finding or creating wait step')
        workflow_step = session.query(AgentWorkflowStep).filter(AgentWorkflowStep.agent_workflow_id == agent_workflow_id, AgentWorkflowStep.unique_id == unique_id).first()
        step_wait = AgentWorkflowStepWait.find_or_create_wait(session=session, step_unique_id=unique_id, description=wait_description, delay=delay)
        if workflow_step is None:
            workflow_step = AgentWorkflowStep(unique_id=unique_id, step_type=step_type, agent_workflow_id=agent_workflow_id)
            session.add(workflow_step)
            session.commit()
        workflow_step.step_type = step_type
        workflow_step.agent_workflow_id = agent_workflow_id
        workflow_step.action_reference_id = step_wait.id
        workflow_step.action_type = 'WAIT_STEP'
        workflow_step.next_steps = []
        session.commit()
        return workflow_step

    @classmethod
    def find_or_create_iteration_workflow_step(cls, session, agent_workflow_id: int, unique_id: str, iteration_workflow_name: str, step_type='NORMAL'):
        if False:
            for i in range(10):
                print('nop')
        ' Find or create a iteration workflow step\n\n        Args:\n            session: db session\n            agent_workflow_id: id of the agent workflow\n            unique_id: unique id of the step\n            iteration_workflow_name: name of the iteration workflow\n            step_type: type of the step\n\n        Returns:\n            AgentWorkflowStep.\n        '
        workflow_step = session.query(AgentWorkflowStep).filter(AgentWorkflowStep.agent_workflow_id == agent_workflow_id, AgentWorkflowStep.unique_id == unique_id).first()
        iteration_workflow = IterationWorkflow.find_workflow_by_name(session, iteration_workflow_name)
        if workflow_step is None:
            workflow_step = AgentWorkflowStep(unique_id=unique_id, step_type=step_type, agent_workflow_id=agent_workflow_id)
            session.add(workflow_step)
            session.commit()
        workflow_step.step_type = step_type
        workflow_step.agent_workflow_id = agent_workflow_id
        workflow_step.action_reference_id = iteration_workflow.id
        workflow_step.action_type = 'ITERATION_WORKFLOW'
        workflow_step.next_steps = []
        session.commit()
        return workflow_step

    @classmethod
    def add_next_workflow_step(cls, session, current_agent_step_id: int, next_step_id: int, step_response: str='default'):
        if False:
            print('Hello World!')
        ' Add Next workflow steps in the next_steps column\n\n        Args:\n            session: db session\n            current_agent_step_id: id of the current agent step\n            next_step_id: id of the next agent step\n            step_response: response of the current step\n\n        '
        next_unique_id = '-1'
        if next_step_id != -1:
            next_workflow_step = AgentWorkflowStep.find_by_id(session, next_step_id)
            next_unique_id = next_workflow_step.unique_id
        current_step = session.query(AgentWorkflowStep).filter(AgentWorkflowStep.id == current_agent_step_id).first()
        next_steps = json.loads(json.dumps(current_step.next_steps))
        existing_steps = [step for step in next_steps if step['step_id'] == next_unique_id]
        if existing_steps:
            existing_steps[0]['step_response'] = step_response
            current_step.next_steps = next_steps
        else:
            next_steps.append({'step_response': str(step_response), 'step_id': str(next_unique_id)})
            current_step.next_steps = next_steps
        session.commit()
        return current_step

    @classmethod
    def fetch_default_next_step(cls, session, current_agent_step_id: int):
        if False:
            return 10
        ' Fetches the default next step\n\n        Args:\n            session: db session\n            current_agent_step_id: id of the current agent step\n        '
        current_step = AgentWorkflowStep.find_by_id(session, current_agent_step_id)
        next_steps = current_step.next_steps
        default_steps = [step for step in next_steps if step['step_response'] == 'default']
        if default_steps:
            return AgentWorkflowStep.find_by_unique_id(session, default_steps[0]['step_id'])
        return None

    @classmethod
    def fetch_next_step(cls, session, current_agent_step_id: int, step_response: str):
        if False:
            print('Hello World!')
        ' Fetch the next step based on the step response\n\n        Args:\n            session: db session\n            current_agent_step_id: id of the current agent step\n            step_response: response of the current step\n        '
        current_step = AgentWorkflowStep.find_by_id(session, current_agent_step_id)
        next_steps = current_step.next_steps
        matching_steps = [step for step in next_steps if str(step['step_response']).lower() == step_response.lower()]
        if matching_steps:
            if str(matching_steps[0]['step_id']) == '-1':
                return 'COMPLETE'
            return AgentWorkflowStep.find_by_unique_id(session, matching_steps[0]['step_id'])
        logger.info(f'Could not find next step for step_id: {current_agent_step_id} and step_response: {step_response}')
        default_steps = [step for step in next_steps if str(step['step_response']).lower() == 'default']
        if default_steps:
            if str(default_steps[0]['step_id']) == '-1':
                return 'COMPLETE'
            return AgentWorkflowStep.find_by_unique_id(session, default_steps[0]['step_id'])
        return None