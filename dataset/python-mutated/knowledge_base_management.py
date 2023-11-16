"""Dialogflow API Python sample showing how to manage Knowledge bases.

Examples:
  python knowledge_base_management.py -h
  python knowledge_base_management.py --project-id PROJECT_ID   list
  python knowledge_base_management.py --project-id PROJECT_ID   create --display-name DISPLAY_NAME
  python knowledge_base_management.py --project-id PROJECT_ID   get --knowledge-base-id knowledge_base_id
  python knowledge_base_management.py --project-id PROJECT_ID   delete --knowledge-base-id knowledge_base_id
"""
import argparse

def create_knowledge_base(project_id, display_name):
    if False:
        i = 10
        return i + 15
    'Creates a Knowledge base.\n\n    Args:\n        project_id: The GCP project linked with the agent.\n        display_name: The display name of the Knowledge base.'
    from google.cloud import dialogflow_v2beta1 as dialogflow
    client = dialogflow.KnowledgeBasesClient()
    project_path = client.common_project_path(project_id)
    knowledge_base = dialogflow.KnowledgeBase(display_name=display_name)
    response = client.create_knowledge_base(parent=project_path, knowledge_base=knowledge_base)
    print('Knowledge Base created:\n')
    print('Display Name: {}\n'.format(response.display_name))
    print('Name: {}\n'.format(response.name))

def get_knowledge_base(project_id, knowledge_base_id):
    if False:
        print('Hello World!')
    'Gets a specific Knowledge base.\n\n    Args:\n        project_id: The GCP project linked with the agent.\n        knowledge_base_id: Id of the Knowledge base.'
    from google.cloud import dialogflow_v2beta1 as dialogflow
    client = dialogflow.KnowledgeBasesClient()
    knowledge_base_path = client.knowledge_base_path(project_id, knowledge_base_id)
    response = client.get_knowledge_base(name=knowledge_base_path)
    print('Got Knowledge Base:')
    print(' - Display Name: {}'.format(response.display_name))
    print(' - Name: {}'.format(response.name))
    return response

def delete_knowledge_base(project_id, knowledge_base_id):
    if False:
        for i in range(10):
            print('nop')
    'Deletes a specific Knowledge base.\n\n    Args:\n        project_id: The GCP project linked with the agent.\n        knowledge_base_id: Id of the Knowledge base.'
    from google.cloud import dialogflow_v2beta1 as dialogflow
    client = dialogflow.KnowledgeBasesClient()
    knowledge_base_path = client.knowledge_base_path(project_id, knowledge_base_id)
    client.delete_knowledge_base(name=knowledge_base_path)
    print('Knowledge Base deleted.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-id', help='Project/agent id.', required=True)
    subparsers = parser.add_subparsers(dest='command')
    list_parser = subparsers.add_parser('list', help='List all Knowledge bases that belong to the project.')
    create_parser = subparsers.add_parser('create', help='Create a new Knowledge base.')
    create_parser.add_argument('--display-name', help='A name of the Knowledge base, used for display purpose, can not be used to identify the Knowledge base.', default=str(''))
    get_parser = subparsers.add_parser('get', help='Get a Knowledge base by its id.')
    get_parser.add_argument('--knowledge-base-id', help='The id of the Knowledge base.', required=True)
    delete_parser = subparsers.add_parser('delete', help='Delete a Knowledge base by its id.')
    delete_parser.add_argument('--knowledge-base-id', help='The id of the Knowledge base you want to delete.', required=True)
    args = parser.parse_args()
    if args.command == 'create':
        create_knowledge_base(args.project_id, args.display_name)
    elif args.command == 'get':
        get_knowledge_base(args.project_id, args.knowledge_base_id)
    elif args.command == 'delete':
        delete_knowledge_base(args.project_id, args.knowledge_base_id)