from shortGPT.gpt import gpt_utils
import random
import json

def generateRedditPostMetadata(title):
    if False:
        for i in range(10):
            print('nop')
    name = generateUsername()
    if title and title[0] == '"':
        title = title.replace('"', '')
    n_months = random.randint(1, 11)
    header = f'{name} - {n_months} months ago'
    n_comments = random.random() * 10 + 2
    n_upvotes = n_comments * (1.2 + random.random() * 2.5)
    return (title, header, f'{n_comments:.1f}k', f'{n_upvotes:.1f}k')

def getInterestingRedditQuestion():
    if False:
        for i in range(10):
            print('nop')
    (chat, system) = gpt_utils.load_local_yaml_prompt('prompt_templates/reddit_generate_question.yaml')
    return gpt_utils.gpt3Turbo_completion(chat_prompt=chat, system=system, temp=1.08)

def createRedditScript(question):
    if False:
        while True:
            i = 10
    (chat, system) = gpt_utils.load_local_yaml_prompt('prompt_templates/reddit_generate_script.yaml')
    chat = chat.replace('<<QUESTION>>', question)
    result = 'Reddit, ' + question + ' ' + gpt_utils.gpt3Turbo_completion(chat_prompt=chat, system=system, temp=1.08)
    return result

def getRealisticness(text):
    if False:
        while True:
            i = 10
    (chat, system) = gpt_utils.load_local_yaml_prompt('prompt_templates/reddit_filter_realistic.yaml')
    chat = chat.replace('<<INPUT>>', text)
    while True:
        try:
            result = gpt_utils.gpt3Turbo_completion(chat_prompt=chat, system=system, temp=1)
            return json.loads(result)['score']
        except Exception as e:
            print('Error in getRealisticness', e.args[0])

def getQuestionFromThread(text):
    if False:
        for i in range(10):
            print('nop')
    if text.find('Reddit, ') < 15 and 10 < text.find('?') < 100:
        question = text.split('?')[0].replace('Reddit, ', '').strip().capitalize()
    else:
        (chat, system) = gpt_utils.load_local_yaml_prompt('prompt_templates/reddit_filter_realistic.yaml')
        chat = chat.replace('<<STORY>>', text)
        question = gpt_utils.gpt3Turbo_completion(chat_prompt=chat, system=system).replace('\n', '')
        question = question.replace('"', '').replace('?', '')
    return question

def generateUsername():
    if False:
        for i in range(10):
            print('nop')
    (chat, system) = gpt_utils.load_local_yaml_prompt('prompt_templates/reddit_username.yaml')
    return gpt_utils.gpt3Turbo_completion(chat_prompt=chat, system=system, temp=1.2).replace('u/', '')