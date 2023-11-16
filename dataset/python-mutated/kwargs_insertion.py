import inspect
keyword_args = ['Keyword Arguments:', '    read_timeout ({read_timeout_type}, optional): Value to pass to         :paramref:`telegram.request.BaseRequest.post.read_timeout`. Defaults to         {read_timeout}.', '    write_timeout (:obj:`float` | :obj:`None`, optional): Value to pass to         :paramref:`telegram.request.BaseRequest.post.write_timeout`. Defaults to         {write_timeout}.', '    connect_timeout (:obj:`float` | :obj:`None`, optional): Value to pass to         :paramref:`telegram.request.BaseRequest.post.connect_timeout`. Defaults to         :attr:`~telegram.request.BaseRequest.DEFAULT_NONE`.', '    pool_timeout (:obj:`float` | :obj:`None`, optional): Value to pass to         :paramref:`telegram.request.BaseRequest.post.pool_timeout`. Defaults to         :attr:`~telegram.request.BaseRequest.DEFAULT_NONE`.', '    api_kwargs (:obj:`dict`, optional): Arbitrary keyword arguments        to be passed to the Telegram API.', '']
write_timeout_sub = [':attr:`~telegram.request.BaseRequest.DEFAULT_NONE`', '``20``']
read_timeout_sub = [':attr:`~telegram.request.BaseRequest.DEFAULT_NONE`', '``2``. :paramref:`timeout` will be added to this value']
read_timeout_type = [':obj:`float` | :obj:`None`', ':obj:`float`']

def find_insert_pos_for_kwargs(lines: list[str]) -> int:
    if False:
        return 10
    'Finds the correct position to insert the keyword arguments and returns the index.'
    for (idx, value) in reversed(list(enumerate(lines))):
        if value.startswith('Returns'):
            return idx
    else:
        return False

def is_write_timeout_20(obj: object) -> int:
    if False:
        while True:
            i = 10
    'inspects the default value of write_timeout parameter of the bot method.'
    sig = inspect.signature(obj)
    return 1 if sig.parameters['write_timeout'].default == 20 else 0

def check_timeout_and_api_kwargs_presence(obj: object) -> int:
    if False:
        print('Hello World!')
    'Checks if the method has timeout and api_kwargs keyword only parameters.'
    sig = inspect.signature(obj)
    params_to_check = ('read_timeout', 'write_timeout', 'connect_timeout', 'pool_timeout', 'api_kwargs')
    return all((param in sig.parameters and sig.parameters[param].kind == inspect.Parameter.KEYWORD_ONLY for param in params_to_check))