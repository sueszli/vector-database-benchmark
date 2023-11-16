"""Utils module for calculating embeddings or completion for text."""
from typing import List, Sequence
from tqdm import tqdm

def call_open_ai_completion_api(inputs: Sequence[str], max_tokens=200, batch_size=20, model: str='text-davinci-003', temperature: float=0.5) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Call the open ai completion api with the given inputs batch by batch.\n\n    Parameters\n    ----------\n    inputs : Sequence[str]\n        The inputs to send to the api.\n    max_tokens : int, default 200\n        The maximum number of tokens to return for each input.\n    batch_size : int, default 20\n        The number of inputs to send in each batch.\n    model : str, default 'text-davinci-003'\n        The model to use for the question answering task. For more information about the models, see:\n        https://beta.openai.com/docs/api-reference/models\n    temperature : float, default 0.5\n        The temperature to use for the question answering task. For more information about the temperature, see:\n        https://beta.openai.com/docs/api-reference/completions/create-completion\n\n    Returns\n    -------\n    List[str]\n        The answers for the questions.\n    "
    try:
        import openai
    except ImportError as e:
        raise ImportError('question_answering_open_ai requires the openai python package. To get it, run "pip install openai".') from e
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
    def _get_answers_with_backoff(questions_in_context):
        if False:
            i = 10
            return i + 15
        return openai.Completion.create(engine=model, prompt=questions_in_context, max_tokens=max_tokens, temperature=temperature)
    answers = []
    for sub_list in tqdm([inputs[x:x + batch_size] for x in range(0, len(inputs), batch_size)], desc=f'Calculating Responses (Total of {len(inputs)})'):
        open_ai_responses = _get_answers_with_backoff(sub_list)
        choices = sorted(open_ai_responses['choices'], key=lambda x: x['index'])
        answers = answers + [choice['text'] for choice in choices]
    return answers