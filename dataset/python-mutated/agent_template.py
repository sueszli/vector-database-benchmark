import json
import requests
from sqlalchemy import Column, Integer, String, Text
from superagi.lib.logger import logger
from superagi.models.agent_template_config import AgentTemplateConfig
from superagi.models.workflows.agent_workflow import AgentWorkflow
from superagi.models.base_model import DBBaseModel
from superagi.models.workflows.iteration_workflow import IterationWorkflow
marketplace_url = 'https://app.superagi.com/api/'

class AgentTemplate(DBBaseModel):
    """
    Preconfigured agent templates that can be used to create agents.

    Attributes:
        id (int): The unique identifier of the agent template.
        organisation_id (int): The organization ID of the user or -1 if the template is public.
        agent_workflow_id (int): The identifier of the workflow that the agent will use.
        name (str): The name of the agent template.
        description (str): The description of the agent template.
        marketplace_template_id (int): The ID of the template in the marketplace.
    """
    __tablename__ = 'agent_templates'
    id = Column(Integer, primary_key=True)
    organisation_id = Column(Integer)
    agent_workflow_id = Column(Integer)
    name = Column(String)
    description = Column(Text)
    marketplace_template_id = Column(Integer)

    def __repr__(self):
        if False:
            print('Hello World!')
        '\n        Returns a string representation of the AgentTemplate object.\n\n        Returns:\n            str: String representation of the AgentTemplate.\n        '
        return f"AgentTemplate(id={self.id}, name='{self.name}', description='{self.description}')"

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        '\n        Converts the AgentTemplate object to a dictionary.\n\n        Returns:\n            dict: Dictionary representation of the AgentTemplate.\n        '
        return {'id': self.id, 'name': self.name, 'description': self.description}

    def to_json(self):
        if False:
            i = 10
            return i + 15
        '\n        Converts the AgentTemplate object to a JSON string.\n\n        Returns:\n            str: JSON string representation of the AgentTemplate.\n        '
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_data):
        if False:
            while True:
                i = 10
        '\n        Creates an AgentTemplate object from a JSON string.\n\n        Args:\n            json_data (str): JSON string representing the AgentTemplate.\n\n        Returns:\n            AgentTemplate: AgentTemplate object created from the JSON string.\n        '
        data = json.loads(json_data)
        return cls(id=data['id'], name=data['name'], description=data['description'])

    @classmethod
    def main_keys(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the main keys for fetching agent templates.\n\n        Returns:\n            list: List of main keys.\n        '
        keys_to_fetch = ['goal', 'instruction', 'constraints', 'tools', 'exit', 'iteration_interval', 'model', 'permission_type', 'LTM_DB', 'max_iterations', 'knowledge']
        return keys_to_fetch

    @classmethod
    def fetch_marketplace_list(cls, search_str, page):
        if False:
            print('Hello World!')
        '\n        Fetches a list of agent templates from the marketplace.\n\n        Args:\n            search_str (str): The search string to filter agent templates.\n            page (int): The page number of the result set.\n\n        Returns:\n            list: List of agent templates fetched from the marketplace.\n        '
        headers = {'Content-Type': 'application/json'}
        response = requests.get(marketplace_url + 'agent_templates/marketplace/list?search=' + search_str + '&page=' + str(page), headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return []

    @classmethod
    def fetch_marketplace_detail(cls, agent_template_id):
        if False:
            return 10
        '\n        Fetches the details of an agent template from the marketplace.\n\n        Args:\n            agent_template_id (int): The ID of the agent template.\n\n        Returns:\n            dict: Details of the agent template fetched from the marketplace.\n        '
        headers = {'Content-Type': 'application/json'}
        response = requests.get(marketplace_url + 'agent_templates/marketplace/template_details/' + str(agent_template_id), headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {}

    @classmethod
    def clone_agent_template_from_marketplace(cls, db, organisation_id: int, agent_template_id: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clones an agent template from the marketplace and saves it in the database.\n\n        Args:\n            db: The database object.\n            organisation_id (int): The organization ID.\n            agent_template_id (int): The ID of the agent template in the marketplace.\n\n        Returns:\n            AgentTemplate: The cloned agent template object.\n        '
        agent_template = AgentTemplate.fetch_marketplace_detail(agent_template_id)
        agent_workflow = db.session.query(AgentWorkflow).filter(AgentWorkflow.name == agent_template['agent_workflow_name']).first()
        logger.info('agent_workflow:' + str(agent_template['agent_workflow_name']))
        if not agent_workflow:
            workflow_id = AgentTemplate.fetch_iteration_agent_template_mapping(db.session, agent_template['agent_workflow_name'])
            agent_workflow = db.session.query(AgentWorkflow).filter(AgentWorkflow.id == workflow_id).first()
        template = AgentTemplate(organisation_id=organisation_id, agent_workflow_id=agent_workflow.id, name=agent_template['name'], description=agent_template['description'], marketplace_template_id=agent_template['id'])
        db.session.add(template)
        db.session.commit()
        db.session.flush()
        agent_configurations = []
        for (key, value) in agent_template['configs'].items():
            agent_configurations.append(AgentTemplateConfig(agent_template_id=template.id, key=key, value=str(value['value'])))
        db.session.add_all(agent_configurations)
        db.session.commit()
        db.session.flush()
        return template

    @classmethod
    def fetch_iteration_agent_template_mapping(cls, session, name):
        if False:
            while True:
                i = 10
        if name == 'Fixed Task Queue':
            agent_workflow = AgentWorkflow.find_by_name(session, 'Fixed Task Workflow')
            return agent_workflow.id
        if name == 'Maintain Task Queue':
            agent_workflow = AgentWorkflow.find_by_name(session, 'Dynamic Task Workflow')
            return agent_workflow.id
        if name == "Don't Maintain Task Queue" or name == 'Goal Based Agent':
            agent_workflow = AgentWorkflow.find_by_name(session, 'Goal Based Workflow')
            return agent_workflow.id

    @classmethod
    def eval_agent_config(cls, key, value):
        if False:
            print('Hello World!')
        '\n        Evaluates the value of an agent configuration key.\n\n        Args:\n            key (str): The key of the agent configuration.\n            value (str): The value of the agent configuration.\n\n        Returns:\n            object: The evaluated value of the agent configuration.\n        '
        if key in ['name', 'description', 'exit', 'model', 'permission_type', 'LTM_DB']:
            return value
        elif key in ['project_id', 'memory_window', 'max_iterations', 'iteration_interval', 'knowledge']:
            if value is not None and value != 'None':
                return int(value)
            else:
                return None
        elif key == 'goal' or key == 'constraints' or key == 'instruction':
            return eval(value)
        elif key == 'tools':
            return [str(x) for x in eval(value)]