"""Creates a chat game as an OpenSpiel Environment."""
from typing import Any, Callable, Dict, OrderedDict, List, Tuple, Union
from absl import logging
import numpy as np
from open_spiel.python.games.chat_games import chat_game_base
from open_spiel.python.games.chat_games.configs import config_fixed_mock
from open_spiel.python.games.chat_games.configs import config_rnd_mock
from open_spiel.python.games.chat_games.envs.observations import utils as observation_utils
from open_spiel.python.games.chat_games.envs.payoffs import utils as payoff_utils
from open_spiel.python.games.chat_games.envs.termination import utils as term_utils
from open_spiel.python.games.chat_games.envs.utils import header as header_utils
from open_spiel.python.games.chat_games.utils import test_utils as chat_test_utils
import pyspiel
GAME_TYPE = pyspiel.GameType(short_name='chat_game', long_name='Chat Game', utility=pyspiel.GameType.Utility.GENERAL_SUM, provides_information_state_string=False, provides_information_state_tensor=False, **chat_game_base.GAME_TYPE_KWARGS)

class ChatGameObserver(chat_game_base.ChatGameObserverBase):
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def _build_str_to_info_state(self) -> bool:
        if False:
            while True:
                i = 10
        'Initializes map from str to infostate. Returns True if successful.'
        return True

    def _info_state(self, input_text: str, obs_size: int) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Returns a len-obs_size np.ndarray given an input string and obs_size.'
        if not self._str_to_info_state_built:
            raise ValueError('String to info state mapping not built!')
        del input_text
        return np.zeros(obs_size, dtype=np.int32)

class ChatGame(chat_game_base.BaseChatGame):
    """Chat game."""

    def __init__(self, params: Dict[str, Any]=chat_game_base.DEFAULT_PARAMS):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      params: dict, parameter dict with the following keys\n\n        num_distinct_actions- int, # of actions at each info set\n        num_llm_seeds- int, # of seeds to use for generating LLM response\n        num_players- int, # of speakers (action: recipient) on the message chain\n        min_utility- float, minimum utility any player can attain\n        max_utility- float, maximum utility any player can attain\n        num_max_replies- int, total # of messages each player can send in an\n          episode\n    '
        self._game_loaded = False
        super().__init__(params)
        super(chat_game_base.BaseChatGame, self).__init__(GAME_TYPE, self.game_info, params or dict())

    def load_chat_game(self, llm_type: chat_test_utils.TestLLM, observations: List[observation_utils.Observation], vectorize: ..., header: header_utils.Header, payoffs: List[payoff_utils.Payoff], aggregate_payoffs: Callable[[List[int]], float]=np.mean, given_names: Union[List[str], None]=None, given_llm_seeds: Union[List[int], None]=None, given_prompt_actions: Union[OrderedDict[str, List[str]], None]=None, given_private_info: Union[OrderedDict[str, List[str]], None]=None, initial_scenario: Union[Any, None]=None, num_names: int=2, num_prompt_actions: Tuple[int, ...]=(4,), num_private_info: Tuple[int, ...]=(4,), examples_names: Union[List[str], None]=None, examples_prompt_actions: Union[OrderedDict[str, List[str]], None]=None, examples_private_info: Union[OrderedDict[str, List[str]], None]=None, examples_scenarios: Union[List[Any], None]=None, llm_list_suffix: str='Continue the list from here.', llm_termination_prompt: Union[term_utils.Termination, None]=None, seed: Union[int, None]=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n    Args:\n      llm_type: item of enum type chat_test_utils.TestLLM\n      observations: List of Observation items used for prompting llms to extract\n        observations (string features) from dialogues\n      vectorize: converts any length string into a length obs_size vector\n\n      header: List of Header items used for prompting llms to take actions\n        (construct messages) based on latent action variables and private\n        information\n\n      payoffs: list of Payoff items used for constructing queries and scoring\n        dialogue for each agent\n      aggregate_payoffs: function that maps from vector to nonnegative scalar\n      \n      given_names: list of strings representing names of players\n      given_llm_seeds: list of ints to seed llm with to generate each message\n      given_prompt_actions: ordered dict mapping action_keys\n        (see envs/utils/header) to list of strings representing the set of\n        available prompt actions (e.g., personalities or msg tones). Overrides\n        examples_prompt_actions.\n      given_private_info: ordered dict mapping info_keys\n        (see envs/utils/header) to length-[num_players] list of strings\n        representing the private information available to each player (e.g.,\n        inventory / valuations of fruits). Overrides examples_private_info.\n      initial_scenario: Scenario items representing an initial message\n\n      num_names: int, # of names to generate (can be greater than # of players)\n      num_prompt_actions: tuple of int, # of prompts to consider for each\n        action_key (i.e., size of action space for each prompt action)\n      num_private_info: tuple of int, # of private info states to consider for\n        each info_key\n      \n      examples_names: list of strings representing examples of names of players\n      examples_prompt_actions: ordered dict mapping action_keys\n        (see envs/utils/header) to list of strings representing examples of\n        prompt actions (e.g., personalities or msg tones).\n      examples_private_info: ordered dict mapping info_keys\n        (see envs/utils/header) to list of strings representing examples of\n        private information available to players (e.g., inventory / valuations\n        of fruits). Overrides examples_private_info.\n      examples_scenarios: list of Scenario items used for meta-generating new\n        scenarios\n      \n      llm_list_suffix: str, gets appended to a prompt to induce an llm to\n        generate a list of items (different llms like different prompts).\n        chinchilla likes ``, llmit likes `Continue the list from here.`\n      llm_termination_prompt: Termination item w/ [attrs query,\n        obs_trans_postfix, postfix]. llm will be asked to score a binary\n        response `yes`/`no` given query.format(msg=last_msg) to determine\n        whether the episode has reached a terminal state (e.g., deal has been\n        agreed upon). default is empty string in which case llm terminal\n        condition is left unused and episode terminates after\n        num_players * num_max_replies\n\n      seed: int, master seed for experiment (used to generate all subsequent\n        seeds for any random generation)\n    '
        self._llm_type = llm_type
        if self._llm_type == chat_test_utils.TestLLM.MOCK:
            self._lm = chat_test_utils.MockLLM()
        else:
            raise NotImplementedError(f'llm_type {self._llm_type} not available.')
        super()._load_chat_game(observations, vectorize, header, payoffs, aggregate_payoffs, given_names, given_llm_seeds, given_prompt_actions, given_private_info, initial_scenario, num_names, num_prompt_actions, num_private_info, examples_names, examples_prompt_actions, examples_private_info, examples_scenarios, llm_list_suffix, llm_termination_prompt, seed)
        self._game_loaded = True

    def generate_response(self, prompt: str, seed: int, num_output_tokens: Union[int, None]=None) -> str:
        if False:
            return 10
        'Returns LLM generated string given prompt and seed.'
        if self._llm_type == chat_test_utils.TestLLM.MOCK:
            return self._lm.generate_response(prompt, seed, num_output_tokens)
        else:
            raise NotImplementedError(f'llm_type {self._llm_type} not available.')

    def generate_bool(self, prompt: str, seed: int) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns LLM generated boolean given prompt and seed.'
        if self._llm_type == chat_test_utils.TestLLM.MOCK:
            return self._lm.generate_bool(prompt, seed)
        else:
            raise NotImplementedError(f'llm_type {self._llm_type} not available.')

    def make_py_observer(self, iig_obs_type: Union[pyspiel.IIGObservationType, None]=None, params: Union[Dict[str, Any], None]=None) -> ChatGameObserver:
        if False:
            return 10
        'Returns an object used for observing game state.'
        return ChatGameObserver(iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params)

    def new_initial_state(self) -> chat_game_base.ChatGameState:
        if False:
            while True:
                i = 10
        'Generates a new dialogue game.\n\n    Returns:\n      chat_game_base.ChatGameState (see chat_games/chat_game_base.py)\n    '
        if not self._game_loaded:
            if self._num_players == 2:
                config = config_fixed_mock.get_config()
                tones = config.game.given_prompt_actions.values()[0]
                num_prompt_actions = (len(tones),)
            else:
                config = config_rnd_mock.get_config()
                num_prompt_actions = config.game.num_prompt_actions
            self._num_distinct_actions = np.prod(num_prompt_actions + (self._num_players,))
            vectorizer = chat_test_utils.MockVectorizer()
            self.load_chat_game(llm_type=chat_test_utils.TestLLM.MOCK, vectorize=vectorizer.vectorize, seed=1234, **config.game)
            logging.warning('Loading chat_game with default config. Only meant for ' + 'open_spiel testing.')
        return chat_game_base.ChatGameState(self, *super().new_initial_state_specs())
pyspiel.register_game(GAME_TYPE, ChatGame)