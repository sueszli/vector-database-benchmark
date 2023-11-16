import random
import string
from pokemongo_bot import logger
from pokemongo_bot.inventory import player
from pokemongo_bot.base_task import BaseTask
from pokemongo_bot.worker_result import WorkerResult
from pokemongo_bot.human_behaviour import sleep

class CompleteTutorial(BaseTask):
    SUPPORTED_TASK_API_VERSION = 1

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.nickname = self.config.get('nickname', '')
        self.random_nickname = self.config.get('random_nickname', False)
        self.team = self.config.get('team', 0)
        self.tutorial_run = True
        self.team_run = True

    def work(self):
        if False:
            i = 10
            return i + 15
        if self.tutorial_run:
            self.tutorial_run = False
            if not self._check_tutorial_state():
                return WorkerResult.ERROR
        if self.team_run and player()._level >= 5:
            self.team_run = False
            if not self._set_team():
                return WorkerResult.ERROR
        return WorkerResult.SUCCESS

    def _check_tutorial_state(self):
        if False:
            print('Hello World!')
        self._player = self.bot.player_data
        tutorial_state = self._player.get('tutorial_state', [])
        if not 0 in tutorial_state:
            sleep(2)
            if self._set_tutorial_state(0):
                self.logger.info('Completed legal screen')
                tutorial_state = self._player.get('tutorial_state', [])
            else:
                return False
        if not 1 in tutorial_state:
            sleep(7)
            if self._set_avatar():
                if self._set_tutorial_state(1):
                    self.logger.info('Completed avatar selection')
                    tutorial_state = self._player.get('tutorial_state', [])
                else:
                    return False
            else:
                self.logger.error('Error during avatar selection')
                return False
        if not 3 in tutorial_state:
            sleep(10)
            if self._encounter_tutorial():
                self.logger.info('Completed first capture')
            else:
                self.logger.error('Error during first capture')
                return False
        if not 4 in tutorial_state:
            if not self.nickname and (not self.random_nickname):
                self.logger.info('No nickname defined in config')
                return False
            if self.random_nickname:
                min_char = 8
                max_char = 14
                allchar = string.ascii_letters + string.digits
                self.nickname = ''.join((random.choice(allchar) for x in range(random.randint(min_char, max_char))))
            self.logger.info(u'Trying to set {} as nickname'.format(self.nickname))
            sleep(5)
            if self._set_nickname(self.nickname):
                self._set_tutorial_state(4)
                tutorial_state = self._player.get('tutorial_state', [])
            else:
                self.logger.error('Error trying to set nickname')
                return False
        if not 7 in tutorial_state:
            if self._set_tutorial_state(7):
                self.logger.info('Completed first time experience')
            else:
                return False
        return True

    def _encounter_tutorial(self):
        if False:
            while True:
                i = 10
        first_pokemon_id = random.choice([1, 4, 7])
        request = self.bot.api.create_request()
        request.encounter_tutorial_complete(pokemon_id=first_pokemon_id)
        response_dict = request.call()
        try:
            if response_dict['responses']['ENCOUNTER_TUTORIAL_COMPLETE']['result'] == 1:
                return True
            else:
                self.logger.error('Error during encouter tutorial')
                return False
        except KeyError:
            self.logger.error('KeyError during encouter tutorial')
            return False

    def _random_avatar(self):
        if False:
            while True:
                i = 10
        avatar = {}
        avatar['avatar'] = random.randint(0, 1)
        avatar['skin'] = random.randint(1, 3)
        avatar['hair'] = random.randint(1, 5)
        avatar['shirt'] = random.randint(1, 3)
        avatar['pants'] = random.randint(1, 2)
        avatar['hat'] = random.randint(1, 3)
        avatar['shoes'] = random.randint(1, 6)
        avatar['eyes'] = random.randint(1, 4)
        avatar['backpack'] = random.randint(1, 5)
        return avatar

    def _set_avatar(self):
        if False:
            print('Hello World!')
        avatar = self._random_avatar()
        request = self.bot.api.create_request()
        request.set_avatar(player_avatar=avatar)
        response_dict = request.call()
        status = response_dict['responses']['SET_AVATAR']['status']
        try:
            if status == 1:
                return True
            else:
                error_codes = {0: 'UNSET', 1: 'SUCCESS', 2: 'AVATAR_ALREADY_SET', 3: 'FAILURE'}
                self.logger.error('Error during avatar selection : {}'.format(error_codes[status]))
                return False
        except KeyError:
            self.logger.error('KeyError during avatar selection')
            return False

    def _set_nickname(self, nickname):
        if False:
            while True:
                i = 10
        request = self.bot.api.create_request()
        request.claim_codename(codename=nickname)
        response_dict = request.call()
        try:
            result = response_dict['responses']['CLAIM_CODENAME']['status']
            if result == 1:
                self.logger.info(u'Name changed to {}'.format(nickname))
                return True
            else:
                error_codes = {0: 'UNSET', 1: 'SUCCESS', 2: 'CODENAME_NOT_AVAILABLE', 3: 'CODENAME_NOT_VALID', 4: 'CURRENT_OWNER', 5: 'CODENAME_CHANGE_NOT_ALLOWED'}
                self.logger.error(u'Error while changing nickname : {}'.format(error_codes[result]))
                return False
        except KeyError:
            return False

    def _set_tutorial_state(self, completed):
        if False:
            for i in range(10):
                print('nop')
        request = self.bot.api.create_request()
        request.mark_tutorial_complete(tutorials_completed=[completed], send_marketing_emails=False, send_push_notifications=False)
        response_dict = request.call()
        try:
            self._player = response_dict['responses']['MARK_TUTORIAL_COMPLETE']['player_data']
            return response_dict['responses']['MARK_TUTORIAL_COMPLETE']['success']
        except KeyError:
            self.logger.error('KeyError while setting tutorial state')
            return False

    def _set_team(self):
        if False:
            for i in range(10):
                print('nop')
        if self.team == 0:
            return True
        if self.bot.player_data.get('team', 0) != 0:
            self.logger.info(u'Team already picked')
            return True
        sleep(10)
        request = self.bot.api.create_request()
        request.set_player_team(team=self.team)
        response_dict = request.call()
        try:
            result = response_dict['responses']['SET_PLAYER_TEAM']['status']
            if result == 1:
                team_codes = {1: 'Mystic (BLUE)', 2: 'Valor (RED)', 3: 'Instinct (YELLOW)'}
                self.logger.info(u'Picked Team {}.'.format(team_codes[self.team]))
                return True
            else:
                error_codes = {0: 'UNSET', 1: 'SUCCESS', 2: 'TEAM_ALREADY_SET', 3: 'FAILURE'}
                self.logger.error(u'Error while picking team : {}'.format(error_codes[result]))
                return False
        except KeyError:
            return False