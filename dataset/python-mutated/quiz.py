from bridges.python.src.sdk.leon import leon
from bridges.python.src.sdk.types import ActionParams
from ..lib import memory
from typing import Union
groups = [{'name': 'mind', 'a': 'E', 'b': 'I', 'questions': [1, 5, 9, 13, 17]}, {'name': 'energy', 'a': 'S', 'b': 'N', 'questions': [2, 6, 10, 14, 18]}, {'name': 'nature', 'a': 'T', 'b': 'F', 'questions': [3, 7, 11, 15, 19]}, {'name': 'tactics', 'a': 'J', 'b': 'P', 'questions': [4, 8, 12, 16, 20]}]

def run(params: ActionParams) -> None:
    if False:
        i = 10
        return i + 15
    'Loop over the questions and track choices'
    resolvers = params['resolvers']
    choice = None
    letter: Union[memory.Letter, None] = None
    for resolver in resolvers:
        if resolver['name'] == 'form':
            choice = resolver['value']
    if choice is None:
        return leon.answer({'core': {'isInActionLoop': False}})
    (question, choice) = choice.split('_')
    session = memory.get_session()
    current_question = session['current_question']
    next_question = current_question + 1
    for group in groups:
        if current_question in group['questions']:
            letter = group[choice]
    if letter is not None:
        memory.increment_letter_score(letter)
    memory.upsert_session(next_question)
    if current_question == 20:
        session_result = memory.get_session()
        type_arr = []
        for group in groups:
            group_letter = group['a'] if session_result[group['a']] >= session_result[group['b']] else group['b']
            type_arr.append(group_letter)
        final_type = ''.join(type_arr)
        return leon.answer({'key': 'result', 'data': {'type': final_type, 'type_url': final_type.lower()}, 'core': {'isInActionLoop': False}})
    return leon.answer({'key': str(next_question), 'data': {'question': str(next_question)}})