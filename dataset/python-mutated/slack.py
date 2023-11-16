import json
import requests
from pipelines import main_logger

def send_message_to_webhook(message: str, channel: str, webhook: str) -> dict:
    if False:
        print('Hello World!')
    payload = {'channel': f'#{channel}', 'username': 'Connectors CI/CD Bot', 'text': message}
    response = requests.post(webhook, data={'payload': json.dumps(payload)})
    if not response.ok:
        main_logger.error(f'Failed to send message to slack webhook: {response.text}')
    return response