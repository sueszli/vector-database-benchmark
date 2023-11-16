from serpent.game_agent import GameAgent

class SerpentGameAgent(GameAgent):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.frame_handlers['PLAY'] = self.handle_play
        self.frame_handler_setups['PLAY'] = self.setup_play

    def setup_play(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def handle_play(self, game_frame, game_frame_pipeline):
        if False:
            while True:
                i = 10
        pass