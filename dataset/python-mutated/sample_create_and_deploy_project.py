"""
FILE: sample_create_and_deploy_project.py

DESCRIPTION:
    This sample demonstrates how to create and deploy a Qna project.

USAGE:
    python sample_create_and_deploy_project.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_QUESTIONANSWERING_ENDPOINT - the endpoint to your QuestionAnswering resource.
    2) AZURE_QUESTIONANSWERING_KEY - your QuestionAnswering API key.
"""

def sample_create_and_deploy_project():
    if False:
        print('Hello World!')
    import os
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.language.questionanswering.authoring import AuthoringClient
    endpoint = os.environ['AZURE_QUESTIONANSWERING_ENDPOINT']
    key = os.environ['AZURE_QUESTIONANSWERING_KEY']
    client = AuthoringClient(endpoint, AzureKeyCredential(key))
    with client:
        project_name = 'IssacNewton'
        project = client.create_project(project_name=project_name, options={'description': 'biography of Sir Issac Newton', 'language': 'en', 'multilingualResource': True, 'settings': {'defaultAnswer': 'no answer'}})
        print('view created project info:')
        print('\tname: {}'.format(project['projectName']))
        print('\tlanguage: {}'.format(project['language']))
        print('\tdescription: {}'.format(project['description']))
        print('find created project ..')
        qna_projects = client.list_projects()
        for p in qna_projects:
            if p['projectName'] == project_name:
                print('project: {}'.format(p['projectName']))
                print('\tlanguage: {}'.format(p['language']))
                print('\tdescription: {}'.format(p['description']))
        update_sources_poller = client.begin_update_sources(project_name=project_name, sources=[{'op': 'add', 'value': {'displayName': 'Issac Newton Bio', 'sourceUri': 'https://wikipedia.org/wiki/Isaac_Newton', 'sourceKind': 'url'}}])
        sources = update_sources_poller.result()
        print('list project sources')
        for source in sources:
            print('source name: {}'.format(source.get('displayName', 'N/A')))
            print('\tsource: {}'.format(source['source']))
            print('\tsource Uri: {}'.format(source.get('sourceUri', 'N/A')))
            print('\tsource kind: {}'.format(source['sourceKind']))
        deployment_poller = client.begin_deploy_project(project_name=project_name, deployment_name='production')
        deployment = deployment_poller.result()
        print(f"Deployment successfully created under {deployment['deploymentName']}.")
        deployments = client.list_deployments(project_name=project_name)
        print('view project deployments')
        for d in deployments:
            print(d)
if __name__ == '__main__':
    sample_create_and_deploy_project()