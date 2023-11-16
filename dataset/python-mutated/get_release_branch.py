"""Retrieve the branch name from the release PR"""
import requests

def check_for_release_pr(pull):
    if False:
        return 10
    label = pull['head']['label']
    if label.find('release/') != -1:
        return pull['head']['ref']

def get_release_branch():
    if False:
        return 10
    'Retrieve the release branch from the release PR'
    url = 'https://api.github.com/repos/streamlit/streamlit/pulls'
    response = requests.get(url).json()
    for pull in response:
        ref = check_for_release_pr(pull)
        if ref != None:
            return ref

def main():
    if False:
        return 10
    print(get_release_branch())
if __name__ == '__main__':
    main()