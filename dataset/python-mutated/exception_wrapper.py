import logging
import traceback
import gymnasium as gym
logger = logging.getLogger(__name__)

class TooManyResetAttemptsException(Exception):

    def __init__(self, max_attempts: int):
        if False:
            return 10
        super().__init__(f'Reached the maximum number of attempts ({max_attempts}) to reset an environment.')

class ResetOnExceptionWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, max_reset_attempts: int=5):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(env)
        self.max_reset_attempts = max_reset_attempts

    def reset(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        attempt = 0
        while attempt < self.max_reset_attempts:
            try:
                return self.env.reset(**kwargs)
            except Exception:
                logger.error(traceback.format_exc())
                attempt += 1
        else:
            raise TooManyResetAttemptsException(self.max_reset_attempts)

    def step(self, action):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.env.step(action)
        except Exception:
            logger.error(traceback.format_exc())
            return (self.reset(), 0.0, False, {'__terminated__': True})