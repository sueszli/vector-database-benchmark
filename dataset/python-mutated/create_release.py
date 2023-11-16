"""Create a release using Github API"""
import os
import requests

def create_release():
    if False:
        while True:
            i = 10
    'Create a release from the Git Tag'
    tag = os.getenv('GIT_TAG')
    access_token = os.getenv('GH_TOKEN')
    if not tag:
        raise Exception('Unable to retrieve GIT_TAG environment variable')
    url = 'https://api.github.com/repos/streamlit/streamlit/releases'
    header = {'Authorization': f'token {access_token}'}
    response = requests.get(f'{url}/latest', headers=header)
    previous_tag_name = None
    if response.status_code == 200:
        previous_tag_name = response.json()['tag_name']
    else:
        raise Exception(f'Unable get the latest release: {response.text}')
    payload = {'tag_name': tag, 'previous_tag_name': previous_tag_name}
    response = requests.post(f'{url}/generate-notes', json=payload, headers=header)
    body = None
    if response.status_code == 200:
        body = response.json()['body']
    else:
        raise Exception(f'Unable generate the latest release notes: {response.text}')
    payload = {'tag_name': tag, 'name': tag, 'body': body}
    response = requests.post(url, json=payload, headers=header)
    if response.status_code == 201:
        print(f'Successfully created Release {tag}')
    else:
        raise Exception(f'Unable to create release, HTTP response: {response.text}')

def main():
    if False:
        for i in range(10):
            print('nop')
    create_release()
if __name__ == '__main__':
    main()