import json
import boto3
codecommit = boto3.client('codecommit')

def lambda_handler(event, context):
    if False:
        return 10
    references = {reference['ref'] for reference in event['Records'][0]['codecommit']['references']}
    print('References: ' + str(references))
    repository = event['Records'][0]['eventSourceARN'].split(':')[5]
    try:
        response = codecommit.get_repository(repositoryName=repository)
        print('Clone URL: ' + response['repositoryMetadata']['cloneUrlHttp'])
        return response['repositoryMetadata']['cloneUrlHttp']
    except Exception as e:
        print(e)
        print('Error getting repository {}. Make sure it exists and that your repository is in the same region as this function.'.format(repository))
        raise e