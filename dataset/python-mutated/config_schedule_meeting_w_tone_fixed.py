"""A pyspiel config for meta-generated meeting schedule negotiation games.
"""
import collections
from ml_collections import config_dict
from open_spiel.python.games.chat_games.envs.base_envs import schedule_meeting_with_tone_info as env_schedule_meeting_with_tone_info
from open_spiel.python.games.chat_games.envs.observations import summary
from open_spiel.python.games.chat_games.envs.observations import utils as obs_utils
from open_spiel.python.games.chat_games.envs.payoffs import schedule_meeting as payoffs_schedule_meeting
from open_spiel.python.games.chat_games.envs.scenarios.domains import schedule_meeting as scenario_schedule_meeting

def get_config():
    if False:
        while True:
            i = 10
    'Get configuration for chat game.'
    config = config_dict.ConfigDict()
    num_players = 2
    observations = [obs_utils.Observation(summary.PREFIX, summary.POSTFIX) for _ in range(num_players)]
    header = env_schedule_meeting_with_tone_info.HEADER
    payoffs = [payoffs_schedule_meeting.PAYOFF]
    given_prompt_actions = collections.OrderedDict()
    tones = ['calm', 'assertive']
    given_prompt_actions[header.action_keys[0]] = tones
    num_tones = len(tones)
    given_private_info = collections.OrderedDict()
    given_private_info['day_prefs'] = [scenario_schedule_meeting.DAY_PREFS_A, scenario_schedule_meeting.DAY_PREFS_B]
    given_private_info['ooo_days'] = [scenario_schedule_meeting.OOO_A, scenario_schedule_meeting.OOO_B]
    scenario_a = env_schedule_meeting_with_tone_info.Scenario(scenario_schedule_meeting.SCENARIO_A, 'Bob', 'Suzy', scenario_schedule_meeting.OOO_A, scenario_schedule_meeting.DAY_PREFS_A, 'calm')
    params = {'num_distinct_actions': num_players * num_tones, 'num_llm_seeds': 2, 'num_players': num_players, 'min_utility': min([float(p.min) for p in payoffs]), 'max_utility': max([float(p.max) for p in payoffs]), 'num_max_replies': 1}
    config.params = params
    config.game = config_dict.ConfigDict()
    config.game.observations = observations
    config.game.header = header
    config.game.payoffs = payoffs
    config.game.given_prompt_actions = given_prompt_actions
    config.game.num_private_info = (2, 2)
    config.game.given_names = ['Bob', 'Suzy']
    config.game.given_private_info = given_private_info
    config.game.initial_scenario = scenario_a
    config.game.llm_list_suffix = 'Output: '
    return config