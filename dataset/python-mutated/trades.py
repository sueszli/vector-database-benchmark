"""A communication format (substrate) for trades.
"""
from open_spiel.python.games.chat_games.envs.utils import text
CHAR_OPT = '%'
CHAR_MSG = '#'
BLOCK_LEN = 28
SPECIAL_CHARS = (CHAR_OPT, CHAR_MSG)
BLOCK_OPT = CHAR_OPT * BLOCK_LEN
BLOCK_MSG = CHAR_MSG * BLOCK_LEN
PLAIN = '\n\n' + BLOCK_MSG + '\n' + 'Trade Proposal Message:\n' + 'from: {sender}\n' + 'to: {receiver}\n' + BLOCK_MSG + '\n\n'
W_OPTS_PREFIX = '\n\n' + BLOCK_OPT + '\n\n'

def strip_msg(msg: str, terminal_str: str='') -> str:
    if False:
        return 10
    return text.strip_msg(msg, BLOCK_MSG, BLOCK_OPT, terminal_str)