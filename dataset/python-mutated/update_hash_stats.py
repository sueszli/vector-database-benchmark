from datetime import datetime, timedelta
from pokemongo_bot.base_task import BaseTask
from pokemongo_bot.tree_config_builder import ConfigException
from pokemongo_bot.worker_result import WorkerResult
from pgoapi.hash_server import HashServer

class UpdateHashStats(BaseTask):
    """
    Periodically displays the hash stats in the terminal.
    Time return is UTC format.
    
    Example config :
    {
        "type": "UpdateHashStats",
        "config": {
            "enabled": true,
            "min_interval": 60,
            "stats": ["period", "remaining", "maximum", "expiration"]
        }
    }

    min_interval : The minimum interval at which the stats are displayed,
                   in seconds (defaults to 60 seconds).
                   The update interval cannot be accurate as workers run synchronously.
    stats : An array of stats to display and their display order (implicitly),
            see available stats below (defaults to ["period", "remaining", "maximum", "expiration"]).
    """
    SUPPORTED_TASK_API_VERSION = 1

    def initialize(self):
        if False:
            print('Hello World!')
        self.next_update = None
        self.enabled = self.config.get('enabled', False)
        self.min_interval = self.config.get('min_interval', 60)
        self.displayed_stats = self.config.get('stats', ['period', 'remaining', 'maximum', 'expiration'])
        self.bot.event_manager.register_event('log_hash_stats', parameters='stats')

    def work(self):
        if False:
            print('Hello World!')
        if not self._should_display() and self.enabled:
            return WorkerResult.SUCCESS
        line = self._get_stats_line()
        if not line:
            return WorkerResult.SUCCESS
        self._log_on_terminal(line)
        return WorkerResult.SUCCESS

    def _log_on_terminal(self, stats):
        if False:
            return 10
        '\n        Logs the stats into the terminal using an event.\n        :param stats: The stats to display.\n        :type stats: string\n        :return: Nothing.\n        :rtype: None\n        '
        self.emit_event('log_hash_stats', formatted='{stats}', data={'stats': stats})
        self._compute_next_update()

    def _get_stats_line(self):
        if False:
            i = 10
            return i + 15
        '\n        Generates a stats string with the given player stats according to the configuration.\n        :return: A string containing human-readable stats, ready to be displayed.\n        :rtype: string\n        '
        available_stats = {'period': 'Period: {}'.format(datetime.utcfromtimestamp(HashServer.status.get('period', 0))), 'remaining': 'Remaining: {}'.format(HashServer.status.get('remaining', 0)), 'maximum': 'Maximum: {}'.format(HashServer.status.get('maximum', 0)), 'expiration': 'Expiration: {}'.format(datetime.utcfromtimestamp(HashServer.status.get('expiration', 0)))}

        def get_stat(stat):
            if False:
                while True:
                    i = 10
            "\n            Fetches a stat string from the available stats dictionary.\n            :param stat: The stat name.\n            :type stat: string\n            :return: The generated stat string.\n            :rtype: string\n            :raise: ConfigException: When the provided stat string isn't in the available stats\n            dictionary.\n            "
            if stat not in available_stats:
                raise ConfigException("Stat '{}' isn't available for displaying".format(stat))
            return available_stats[stat]
        line = ' | '.join(map(get_stat, self.displayed_stats))
        return line

    def _should_display(self):
        if False:
            print('Hello World!')
        '\n        Returns a value indicating whether the stats should be displayed.\n        :return: True if the stats should be displayed; otherwise, False.\n        :rtype: bool\n        '
        return self.next_update is None or datetime.now() >= self.next_update

    def _compute_next_update(self):
        if False:
            while True:
                i = 10
        '\n        Computes the next update datetime based on the minimum update interval.\n        :return: Nothing.\n        :rtype: None\n        '
        self.next_update = datetime.now() + timedelta(seconds=self.min_interval)