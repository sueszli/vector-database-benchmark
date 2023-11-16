import sys
import os

def main(argv):
    if False:
        print('Hello World!')
    slack_config = os.environ.get('CUPY_CI_SLACK_CONFIG', None)
    gitter_config = os.environ.get('CUPY_CI_GITTER_CONFIG', None)
    subdesc = os.environ.get('FLEXCI_SUB_DESCRIPTION', '')
    if subdesc == '':
        subdesc = os.environ.get('FLEXCI_DESCRIPTION', '(no description)')
    url = os.environ.get('FLEXCI_JOB_URL', '<no url>')
    msg = argv[1]
    body = '*{}:* {}\n\n```\n{}\n```\n'.format(msg, url, subdesc)
    if slack_config is not None:
        from slack_sdk.webhook import WebhookClient
        for hook_url in slack_config.split(','):
            slack = WebhookClient(hook_url)
            slack.send(text=body)
    if gitter_config is not None:
        from gitterpy.client import GitterClient
        (token, rooms) = gitter_config.split(':')
        gitter = GitterClient(token)
        for room in rooms.split(','):
            gitter.messages.send(room, body)
if __name__ == '__main__':
    main(sys.argv)