"""Tests for base environments."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.games.chat_games.envs.base_envs import email_plain
from open_spiel.python.games.chat_games.envs.base_envs import email_with_tone
from open_spiel.python.games.chat_games.envs.base_envs import email_with_tone_info
from open_spiel.python.games.chat_games.envs.base_envs import schedule_meeting_with_info
from open_spiel.python.games.chat_games.envs.base_envs import trade_fruit_with_info
from open_spiel.python.games.chat_games.envs.utils import header

class BaseEnvsTest(parameterized.TestCase):

    @parameterized.parameters([dict(base_env=email_plain), dict(base_env=email_with_tone), dict(base_env=email_with_tone_info), dict(base_env=schedule_meeting_with_info), dict(base_env=trade_fruit_with_info)])
    def test_give_me_a_name(self, base_env):
        if False:
            return 10
        self.assertTrue(header.plain_header_is_valid(base_env.HEADER))
if __name__ == '__main__':
    absltest.main()