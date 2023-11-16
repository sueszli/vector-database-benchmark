"""Test the MCTS Agent."""
from absl.testing import absltest
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import mcts_agent

class MCTSAgentTest(absltest.TestCase):

    def test_tic_tac_toe_episode(self):
        if False:
            for i in range(10):
                print('nop')
        env = rl_environment.Environment('tic_tac_toe', include_full_state=True)
        num_players = env.num_players
        num_actions = env.action_spec()['num_actions']
        mcts_bot = mcts.MCTSBot(env.game, 1.5, 100, mcts.RandomRolloutEvaluator())
        agents = [mcts_agent.MCTSAgent(player_id=idx, num_actions=num_actions, mcts_bot=mcts_bot) for idx in range(num_players)]
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations['current_player']
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])
        for agent in agents:
            agent.step(time_step)
if __name__ == '__main__':
    absltest.main()