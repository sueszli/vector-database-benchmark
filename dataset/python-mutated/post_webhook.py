from hashlib import sha1
from sys import exit
import click
import hmac
import http.client as http_client
import json
import logging
import requests
import urllib3
import uuid

@click.command()
@click.option('--file', required=True, help='File containing the post data.')
@click.option('--key', 'webhook_key', required=True, help='The webhook key for the job template.')
@click.option('--url', required=True, help='The webhook url for the job template (i.e. https://tower.jowestco.net:8043/api/v2/job_templates/637/github/.')
@click.option('--event-type', help='Specific value for Event header, defaults to "issues" for GitHub and "Push Hook" for GitLab')
@click.option('--verbose', is_flag=True, help='Dump HTTP communication for debugging')
@click.option('--insecure', is_flag=True, help='Ignore SSL certs if true')
def post_webhook(file, webhook_key, url, verbose, event_type, insecure):
    if False:
        i = 10
        return i + 15
    '\n    Helper command for submitting POST requests to Webhook endpoints.\n\n    We have two sample webhooks in tools/scripts/webhook_examples for gitlab and github.\n    These or any other file can be pointed to with the --file parameter.\n\n    \x08\n    Additional example webhook events can be found online.\n        For GitLab see:\n        https://docs.gitlab.com/ee/user/project/integrations/webhook_events.html\n\n    \x08\n        For GitHub see:\n        https://docs.github.com/en/developers/webhooks-and-events/webhooks/webhook-events-and-payloads\n\n    \x08\n    For setting up webhooks in AWX see:\n        https://docs.ansible.com/ansible-tower/latest/html/userguide/webhooks.html\n\n    \x08\n    Example usage for GitHub:\n      ./post_webhook.py \\\n        --file webhook_examples/github_push.json \\\n        --url https://tower.jowestco.net:8043/api/v2/job_templates/637/github/ \\\n        --key AvqBR19JDFaLTsbF3p7FmiU9WpuHsJKdHDfTqKXyzv1HtwDGZ8 \\\n        --insecure \\\n        --type github\n    \n    \x08\n    Example usage for GitLab:\n      ./post_webhook.py \\\n        --file webhook_examples/gitlab_push.json \\\n        --url https://tower.jowestco.net:8043/api/v2/job_templates/638/gitlab/ \\\n        --key fZ8vUpfHfb1Dn7zHtyaAsyZC5IHFcZf2a2xiBc2jmrBDptCOL2 \\\n        --insecure \\\n        --type=gitlab \n\n    \x08\n    NOTE: GitLab webhooks are stored in the DB with a UID of the hash of the POST body.\n          After submitting one post GitLab post body a second POST of the same payload \n          can result in a response like: \n              Response code: 202\n              Response body:\n              {\n                  "message": "Webhook previously received, aborting."\n              }\n\n          If you need to test multiple GitLab posts simply change your payload slightly\n\n    '
    if insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    if verbose:
        http_client.HTTPConnection.debuglevel = 1
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger('requests.packages.urllib3')
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True
    with open(file, 'r') as f:
        post_data = json.loads(f.read())
    headers = {'Content-Type': 'application/json'}
    key_bytes = webhook_key.encode('utf-8', 'strict')
    data_bytes = str(json.dumps(post_data)).encode('utf-8', 'strict')
    mac = hmac.new(key_bytes, msg=data_bytes, digestmod=sha1)
    if url.endswith('/github/'):
        headers.update({'X-Hub-Signature': 'sha1={}'.format(mac.hexdigest()), 'X-GitHub-Event': 'issues' if event_type == 'default' else event_type, 'X-GitHub-Delivery': str(uuid.uuid4())})
    elif url.endswith('/gitlab/'):
        mac = hmac.new(key_bytes, msg=data_bytes, digestmod=sha1)
        headers.update({'X-GitLab-Event': 'Push Hook' if event_type == 'default' else event_type, 'X-GitLab-Token': webhook_key})
    else:
        click.echo('This utility only knows how to support URLs that end in /github/ or /gitlab/.')
        exit(250)
    r = requests.post(url, data=json.dumps(post_data), headers=headers, verify=not insecure)
    if not verbose:
        click.echo('Response code: {}'.format(r.status_code))
        click.echo('Response body:')
    try:
        click.echo(json.dumps(r.json(), indent=4))
    except:
        click.echo(r.text)
if __name__ == '__main__':
    post_webhook()