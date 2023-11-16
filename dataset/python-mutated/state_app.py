import arrow
import jesse.helpers as jh

class AppState:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.time = arrow.utcnow().int_timestamp * 1000
        self.starting_time = None
        self.daily_balance = []
        self.total_open_trades = 0
        self.total_open_pl = 0
        self.total_liquidations = 0
        self.session_id = ''
        self.session_info = {}

    def set_session_id(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Generated and sets session_id. Used to prevent overriding of the session_id\n        '
        if self.session_id == '':
            self.session_id = jh.generate_unique_id()