import requests
from ..utils.convert_to_openai_messages import convert_to_openai_messages

def get_relevant_procedures_string(messages):
    if False:
        while True:
            i = 10
    query = {'query': convert_to_openai_messages(messages)}
    url = 'https://open-procedures.replit.app/search/'
    relevant_procedures = requests.post(url, json=query).json()['procedures']
    relevant_procedures = '[Recommended Procedures]\n' + '\n---\n'.join(relevant_procedures) + '\nIn your plan, include steps and, for relevant deprecation notices, **EXACT CODE SNIPPETS** -- these notices will VANISH once you execute your first line of code, so WRITE THEM DOWN NOW if you need them.'
    return relevant_procedures