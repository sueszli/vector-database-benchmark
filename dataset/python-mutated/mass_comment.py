"""Script for mass-commenting Jenkins test triggers on a Beam PR."""
import os
import requests
import socket
import time

def executeGHGraphqlQuery(accessToken, query):
    if False:
        while True:
            i = 10
    'Runs graphql query on GitHub.'
    url = 'https://api.github.com/graphql'
    headers = {'Authorization': 'Bearer %s' % accessToken}
    r = requests.post(url=url, json={'query': query}, headers=headers)
    return r.json()

def getSubjectId(accessToken, prNumber):
    if False:
        i = 10
        return i + 15
    query = '\nquery FindPullRequestID {\n  repository(owner:"apache", name:"beam") {\n    pullRequest(number:%s) {\n      id\n    }\n  }\n}\n' % prNumber
    response = executeGHGraphqlQuery(accessToken, query)
    return response['data']['repository']['pullRequest']['id']

def addPrComment(accessToken, subjectId, commentBody):
    if False:
        print('Hello World!')
    'Adds a pr comment to the PR defined by subjectId'
    query = '\nmutation AddPullRequestComment {\n  addComment(input:{subjectId:"%s",body: "%s"}) {\n    commentEdge {\n        node {\n        createdAt\n        body\n      }\n    }\n    subject {\n      id\n    }\n  }\n}\n' % (subjectId, commentBody)
    return executeGHGraphqlQuery(accessToken, query)

def getPrStatuses(accessToken, prNumber):
    if False:
        i = 10
        return i + 15
    query = '\nquery GetPRChecks {\n  repository(name: "beam", owner: "apache") {\n    pullRequest(number: %s) {\n      commits(last: 1) {\n        nodes {\n          commit {\n            status {\n              contexts {\n                targetUrl\n                context\n              }\n            }\n          }\n        }\n      }\n    }\n  }\n}\n' % prNumber
    return executeGHGraphqlQuery(accessToken, query)

def postComments(accessToken, subjectId, commentsToAdd):
    if False:
        print('Hello World!')
    '\n  Main workhorse method. Posts comments to GH.\n  '
    for comment in commentsToAdd:
        jsonData = addPrComment(accessToken, subjectId, comment[0])
        print(jsonData)
        time.sleep(30)

def probeGitHubIsUp():
    if False:
        while True:
            i = 10
    '\n  Returns True if GitHub responds to simple queries. Else returns False.\n  '
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('github.com', 443))
    return True if result == 0 else False

def getRemainingComments(accessToken, pr, initialComments):
    if False:
        i = 10
        return i + 15
    '\n  Filters out the comments that already have statuses associated with them from initial comments\n  '
    queryResult = getPrStatuses(accessToken, pr)
    pull = queryResult['data']['repository']['pullRequest']
    commit = pull['commits']['nodes'][0]['commit']
    check_urls = str(list(map(lambda c: c['targetUrl'], commit['status']['contexts'])))
    remainingComments = []
    for comment in initialComments:
        if f'/{comment[1]}_Phrase/' not in check_urls and f'/{comment[1]}_PR/' not in check_urls and (f'/{comment[1]}_Commit/' not in check_urls) and (f'/{comment[1]}/' not in check_urls) and ('Sickbay' not in comment[1]):
            print(comment)
            remainingComments.append(comment)
    return remainingComments
if __name__ == '__main__':
    '\n  This script is supposed to be invoked directly.\n  However for testing purposes and to allow importing,\n  wrap work code in module check.\n  '
    print('Started.')
    comments = []
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, 'jenkins_jobs.txt')) as file:
        comments = [line.strip() for line in file if len(line.strip()) > 0]
    for i in range(len(comments)):
        parts = comments[i].split(',')
        comments[i] = (parts[0], parts[1])
    if not probeGitHubIsUp():
        print('GitHub is unavailable, skipping fetching data.')
        exit()
    print('GitHub is available start fetching data.')
    accessToken = input('Enter your Github access token: ')
    pr = input('Enter the Beam PR number to test (e.g. 11403): ')
    subjectId = getSubjectId(accessToken, pr)
    remainingComments = getRemainingComments(accessToken, pr, comments)
    if len(remainingComments) == 0:
        print('Jobs have been started for all comments. If you would like to retry all jobs, create a new commit before running this script.')
    while len(remainingComments) > 0:
        postComments(accessToken, subjectId, remainingComments)
        time.sleep(60)
        remainingComments = getRemainingComments(accessToken, pr, remainingComments)
        if len(remainingComments) > 0:
            print(f'{len(remainingComments)} comments must be reposted because no check has been created for them: {str(remainingComments)}')
            print('Sleeping for 1 hour to allow Jenkins to recover and to give it time to status.')
            for i in range(60):
                time.sleep(60)
                print(f'{i} minutes elapsed, {60 - i} minutes remaining')
        remainingComments = getRemainingComments(accessToken, pr, remainingComments)
        if len(remainingComments) == 0:
            print(f'{len(remainingComments)} comments still must be reposted: {str(remainingComments)}')
            print('Trying to repost comments.')
    print('Done.')